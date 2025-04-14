use std::collections::HashMap;

use mini_zinc::{cgraph::graph::ComputationGraph, op::layer::TensorValue};
use ndarray::ArrayD;

fn main() {
    // let resnet_param = "export/test_resnet/test_resnet.pnnx.param";
    // let resnet_weight = "export/test_resnet/test_resnet.pnnx.bin";
    let resnet_param = "export/resnet/resnet18.pnnx.param";
    let resnet_weight = "export/resnet/resnet18.pnnx.bin";

    let resnet = ComputationGraph::from_pnnx(resnet_param, resnet_weight).unwrap();
    // randomize an input of (1,3,224,224)
    let input = HashMap::from([(0 as usize, TensorValue::Float32(
        ArrayD::from_shape_vec(vec![1, 3, 224, 224], vec![0.0; 1 * 3 * 224 * 224]).unwrap(),
    ))]);
    let result = resnet.compute(&input).unwrap();
    for (k, v) in result {
        println!("{}: {:#?}", k, v);
    }
}