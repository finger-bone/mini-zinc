use anyhow::Result;

use crate::cgraph::{node::CGNodeOp, pnnx_reader::PNNXReaderResult};
use crate::op::dtype::TensorValue;

use super::{node::CGNode, pnnx_weight_reader};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

pub type NodeIdx = usize;
pub type BlobIdx = usize;

pub struct ComputationGraph {
    pub nodes: HashMap<NodeIdx, CGNode>,
    pub input_nodes: Vec<NodeIdx>,
    pub output_nodes: Vec<NodeIdx>,
    pub attribute_nodes: Vec<NodeIdx>,
    pub consumed_by: HashMap<BlobIdx, Vec<NodeIdx>>,
}

impl ComputationGraph {
    pub fn from_pnnx<P: AsRef<Path>>(param_path: P, weight_path: P) -> Result<Self> {
        let param_result = PNNXReaderResult::from_file(param_path)?;
        let (shape_map, dtype_map) = param_result.get_shape_and_dtype_map();
        let weights =
            pnnx_weight_reader::load_pnnx_zip_bin(weight_path, &shape_map, &dtype_map).unwrap();

        let mut nodes = HashMap::<NodeIdx, CGNode>::new();
        let mut input_nodes = Vec::<NodeIdx>::new();
        let mut output_nodes = Vec::<NodeIdx>::new();
        let mut attribute_nodes = Vec::<NodeIdx>::new();
        let mut consumed_by = HashMap::<BlobIdx, Vec<NodeIdx>>::new();

        for (idx, line) in param_result.lines.iter().enumerate() {
            let node = CGNode::from_line(line, &weights).unwrap();
            match &node.op {
                CGNodeOp::Input => {
                    input_nodes.push(idx);
                }
                CGNodeOp::Output => {
                    output_nodes.push(idx);
                }
                CGNodeOp::Attribute(_) => {
                    attribute_nodes.push(idx);
                }
                _ => {}
            }
            nodes.insert(idx, node);
            for input_idx in &line.input_idx_list {
                if let Some(consumed_by_list) = consumed_by.get_mut(input_idx) {
                    consumed_by_list.push(idx);
                } else {
                    consumed_by.insert(*input_idx, vec![idx]);
                }
            }
        }

        Ok(Self {
            nodes,
            input_nodes,
            output_nodes,
            attribute_nodes,
            consumed_by,
        })
    }
}

impl ComputationGraph {
    /// 计算图的前向计算函数，接收输入张量，返回计算结果
    pub fn compute(
        &mut self,
        inputs: &HashMap<BlobIdx, TensorValue>,
    ) -> Result<HashMap<NodeIdx, TensorValue>> {
        // 存储中间计算结果的张量
        let mut blob_store = HashMap::<BlobIdx, Option<TensorValue>>::new();

        // 跟踪每个blob被消费的剩余次数，用于内存管理
        let mut remaining_feeding_times_counter = self.initialize_feeding_counter();

        // 存储最终结果
        let mut result = HashMap::<NodeIdx, TensorValue>::new();

        // 待处理节点队列，使用Vec而不是HashSet以保持确定性执行顺序
        let mut ready_nodes = Vec::<NodeIdx>::new();
        // 跟踪已计算的节点
        let mut computed = HashSet::<NodeIdx>::new();
        // 跟踪每个节点的未满足依赖数量
        let mut pending_dependencies = self.initialize_dependencies();

        // 处理输入和属性节点
        self.process_input_nodes(
            inputs,
            &mut blob_store,
            &mut pending_dependencies,
            &mut ready_nodes,
        );
        self.process_attribute_nodes(
            &mut blob_store,
            &mut result,
            &mut pending_dependencies,
            &mut ready_nodes,
        );

        // 主计算循环
        self.execute_computation_loop(
            &mut blob_store,
            &mut remaining_feeding_times_counter,
            &mut result,
            &mut ready_nodes,
            &mut computed,
            &mut pending_dependencies,
        )?;

        // 找到所有的 Output 节点的输入 blob，只保留这些 blob
        let valid_output_blobs: Vec<BlobIdx> = self
            .output_nodes
            .iter()
            .flat_map(|node_idx| {
                let node = &self.nodes[node_idx];
                node.inputs.iter().cloned()
            })
            .collect();
        let mut filtered_result = HashMap::<NodeIdx, TensorValue>::new();
        for blob_idx in valid_output_blobs {
            if let Some(Some(tensor)) = blob_store.get(&blob_idx) {
                filtered_result.insert(blob_idx, tensor.clone());
            }
        }
        Ok(filtered_result)
    }

    /// 初始化每个blob的消费计数器
    fn initialize_feeding_counter(&self) -> HashMap<BlobIdx, usize> {
        let counter = self
            .consumed_by
            .iter()
            .map(|(blob_idx, consumed_by_list)| (*blob_idx, consumed_by_list.len()))
            .collect::<HashMap<_, _>>();

        counter
    }

    /// 初始化节点依赖关系
    fn initialize_dependencies(&self) -> HashMap<NodeIdx, usize> {
        let mut dependencies = HashMap::<NodeIdx, usize>::new();

        for (idx, node) in &self.nodes {
            if !matches!(node.op, CGNodeOp::Input | CGNodeOp::Attribute(_)) {
                dependencies.insert(*idx, node.inputs.len());
            }
        }

        dependencies
    }

    /// 处理输入节点
    fn process_input_nodes(
        &self,
        inputs: &HashMap<BlobIdx, TensorValue>,
        blob_store: &mut HashMap<BlobIdx, Option<TensorValue>>,
        pending_dependencies: &mut HashMap<NodeIdx, usize>,
        ready_nodes: &mut Vec<NodeIdx>,
    ) {
        for input_node_idx in &self.input_nodes {
            let node = &self.nodes[input_node_idx];

            for node_output in &node.outputs {
                blob_store.insert(*node_output, Some(inputs[input_node_idx].clone()));

                // 更新依赖于此输入的节点
                if let Some(consumers) = self.consumed_by.get(node_output) {
                    for &consumer in consumers {
                        if let Some(deps) = pending_dependencies.get_mut(&consumer) {
                            *deps -= 1;
                            if *deps == 0 {
                                ready_nodes.push(consumer);
                            }
                        }
                    }
                }
            }
        }
    }

    /// 处理属性节点
    fn process_attribute_nodes(
        &self,
        blob_store: &mut HashMap<BlobIdx, Option<TensorValue>>,
        result: &mut HashMap<NodeIdx, TensorValue>,
        pending_dependencies: &mut HashMap<NodeIdx, usize>,
        ready_nodes: &mut Vec<NodeIdx>,
    ) {
        for attribute_node_idx in &self.attribute_nodes {
            let node = &self.nodes[attribute_node_idx];
            let output_blob = node.outputs.first().unwrap();

            if let CGNodeOp::Attribute(data) = &node.op {
                result.insert(*output_blob, data.clone());
                blob_store.insert(*output_blob, Some(data.clone()));

                // 更新依赖于此属性的节点
                if let Some(consumers) = self.consumed_by.get(output_blob) {
                    for &consumer in consumers {
                        if let Some(deps) = pending_dependencies.get_mut(&consumer) {
                            *deps -= 1;
                            if *deps == 0 {
                                ready_nodes.push(consumer);
                            }
                        }
                    }
                }
            } else {
                panic!("Attribute node should have Attribute op")
            }
        }
    }

    /// 执行主计算循环
    fn execute_computation_loop(
        &mut self,
        blob_store: &mut HashMap<BlobIdx, Option<TensorValue>>,
        remaining_feeding_times_counter: &mut HashMap<BlobIdx, usize>,
        result: &mut HashMap<NodeIdx, TensorValue>,
        ready_nodes: &mut Vec<NodeIdx>,
        computed: &mut HashSet<NodeIdx>,
        pending_dependencies: &mut HashMap<NodeIdx, usize>,
    ) -> Result<()> {
        // let mut iteration = 0;
        while !ready_nodes.is_empty() {
            // iteration += 1;
            let node_idx = ready_nodes.remove(0);

            // 跳过已计算的节点
            if computed.contains(&node_idx) {
                continue;
            }

            let node = self.nodes.get_mut(&node_idx).unwrap();

            match &mut node.op {
                CGNodeOp::Input | CGNodeOp::Attribute(_) => {
                    panic!("Input node or attribute node should not appear in ready_nodes")
                }
                CGNodeOp::Op(layer) => {
                    // 收集输入张量
                    let input_tensors = node
                        .inputs
                        .iter()
                        .map(|input_idx| blob_store.get(input_idx).unwrap().clone().unwrap())
                        .collect::<Vec<_>>();
                    // println!("node {}", node.name);
                    // for (k, v) in input_tensors.iter().enumerate() {
                    //     println!("input_tensors[{}]: {:?}", k, v);
                    // }
                    // 更新引用计数并释放不再需要的内存
                    for &input_idx in &node.inputs {
                        if let Some(counter) = remaining_feeding_times_counter.get_mut(&input_idx) {
                            *counter -= 1;
                            if *counter == 0 && blob_store.contains_key(&input_idx) {
                                blob_store.remove(&input_idx);
                            }
                        }
                    }

                    // 执行前向计算
                    let output_tensors = layer.forward(&input_tensors)?;

                    // 存储输出并更新依赖图
                    for i in 0..node.outputs.len() {
                        let output_idx = node.outputs[i];
                        blob_store.insert(output_idx, Some(output_tensors[i].clone()));
                        // 更新依赖于此输出的节点
                        if let Some(consumers) = self.consumed_by.get(&output_idx) {
                            for &consumer in consumers {
                                if let Some(deps) = pending_dependencies.get_mut(&consumer) {
                                    *deps -= 1;
                                    if *deps == 0 {
                                        ready_nodes.push(consumer);
                                    }
                                }
                            }
                        }
                    }
                    // for (k, v) in output_tensors.iter().enumerate() {
                    //     println!("output_tensors[{}]: {:?}", k, v);
                    // }
                }
                CGNodeOp::Output => {
                    // 收集输出节点的结果
                    for &input_idx in &node.inputs {
                        if let Some(Some(tensor)) = blob_store.get(&input_idx) {
                            result.insert(input_idx, tensor.clone());
                        }
                    }
                }
            }
            computed.insert(node_idx);
        }

        Ok(())
    }
}
