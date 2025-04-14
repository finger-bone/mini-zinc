use anyhow::Result;

use crate::{
    cgraph::{node::CGNodeOp, pnnx_reader::PNNXReaderResult},
    op::layer::TensorValue,
};

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
        let mut consumed_by = HashMap::<BlobIdx, Vec<NodeIdx>>::new();

        for (idx, line) in param_result.lines.iter().enumerate() {
            let node = CGNode::from_line(line, &weights).unwrap();
            match node.op {
                CGNodeOp::Input => {
                    input_nodes.push(idx);
                }
                CGNodeOp::Output => {
                    output_nodes.push(idx);
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
            consumed_by,
        })
    }
}

impl ComputationGraph {
    pub fn compute(
        &self,
        inputs: &HashMap<BlobIdx, TensorValue>,
    ) -> Result<HashMap<NodeIdx, TensorValue>> {
        let mut blob_store = HashMap::<BlobIdx, Option<TensorValue>>::new();
        let consumed_by = &self.consumed_by;

        let mut remaining_feeding_times_counter = consumed_by
            .iter()
            .map(|(blob_idx, consumed_by_list)| (blob_idx, consumed_by_list.len()))
            .collect::<HashMap<_, _>>();

        let mut result = HashMap::<NodeIdx, TensorValue>::new();

        let mut feeding_to = Vec::<NodeIdx>::new();

        for input_node_idx in &self.input_nodes {
            for node_output in &self.nodes[input_node_idx].outputs {
                blob_store.insert(*node_output, Some(inputs[input_node_idx].clone()));
                feeding_to.extend(consumed_by[input_node_idx].clone());
            }
        }
        let mut computed = HashSet::<NodeIdx>::new();

        // let mut displayed = HashSet::<BlobIdx>::new();

        'outer: while feeding_to.len() > 0 {
            let node_idx = feeding_to.remove(0);
            // println!("feeding_to: {:?}", feeding_to);
            // println!("computed: {:?}", computed);
            // println!("node: {:#?}", self.nodes[&node_idx]);
            // for (k, v) in blob_store.iter() {
            //     if displayed.contains(k) {
            //         continue;
            //     } else {
            //         displayed.insert(*k);
            //     }
            //     println!("blob_store[{}]: {:?}", k, v)
            // }
            // println!("-----");
            if computed.contains(&node_idx) {
                continue 'outer;
            }
            for input_node_idx in &self.nodes.get(&node_idx).unwrap().inputs {
                if blob_store.get(input_node_idx).is_none() {
                    feeding_to.push(node_idx);
                    if feeding_to[0] == node_idx {
                        panic!("Circular dependency detected");
                    }
                    continue 'outer;
                }
            }
            if let Some(node) = self.nodes.get(&node_idx) {
                match &node.op {
                    CGNodeOp::Input => {
                        panic!("Input node should not appear in feeding_to")
                    }
                    CGNodeOp::Op(layer) => {
                        let input_tensors = node
                            .inputs
                            .iter()
                            .map(|input_idx| blob_store.get(input_idx).unwrap().clone().unwrap())
                            .collect::<Vec<_>>();

                        node.inputs.iter().for_each(|input_idx| {
                            let remaining_feeding_times =
                                remaining_feeding_times_counter.get_mut(input_idx).unwrap();
                            *remaining_feeding_times -= 1;
                            if *remaining_feeding_times == 0 {
                                // take and drop the reference
                                if let Some(_) = blob_store.get(input_idx).take() {}
                            }
                        });

                        let output_tensors = layer.forward(&input_tensors).unwrap();
                        for i in 0..node.outputs.len() {
                            let output_idx = node.outputs[i];
                            blob_store.insert(output_idx, Some(output_tensors[i].clone()));
                            feeding_to
                                .extend(consumed_by.get(&output_idx).unwrap().iter().cloned());
                        }
                        computed.insert(node_idx);
                    }
                    CGNodeOp::Output => {
                        for i in 0..node.inputs.len() {
                            let output_idx = node.inputs[i];
                            let output_tensor =
                                blob_store.get(&output_idx).unwrap().clone().unwrap();
                            result.insert(output_idx, output_tensor);
                        }
                        computed.insert(node_idx);
                    }
                }
            }
        }
        Ok(result)
    }
}
