use anyhow::Result;

use crate::cgraph::{node::CGNodeOp, pnnx_reader::PNNXReaderResult};

use super::{node::CGNode, pnnx_weight_reader};
use std::{collections::HashMap, path::Path};

pub type NodeIdx = usize;
pub type BlobIdx = usize;

pub struct ComputationGraph {
    pub nodes: HashMap<NodeIdx, CGNode>,
    pub input_nodes: Vec<NodeIdx>,
    pub output_nodes: Vec<NodeIdx>,
    pub consumed_by: HashMap<BlobIdx, NodeIdx>,
    pub topology: Vec<NodeIdx>,
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
        }

        // let weight_result = load_pnnx_zip_bin(weight_path);

        todo!()
    }
}
