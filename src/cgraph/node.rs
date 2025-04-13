use std::{collections::HashMap, path::Path};

use anyhow::{anyhow, Result};

use crate::{cgraph::{pnnx_reader, pnnx_weight_reader::load_pnnx_zip_bin}, op::layer::Forward};

use super::pnnx_reader::PNNXReaderResult;

pub enum CGNodeOp {
    Input,
    Output,
    Op(Box<dyn Forward>),
}

pub struct CGNode {
    pub idx: usize,
    pub name: String,
    pub op: CGNodeOp,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

pub struct ComputationGraph {
    pub nodes: HashMap<usize, CGNode>,
    pub input_nodes: Vec<usize>,
    pub output_nodes: Vec<usize>,
}

impl ComputationGraph {

    pub fn from_pnnx<P: AsRef<Path>>(
        param_path: P, weight_path: P
    ) -> Result<Self> {
        let param_result = PNNXReaderResult::from_file(param_path)?;

        

        // let weight_result = load_pnnx_zip_bin(weight_path);


        todo!()
    }
}