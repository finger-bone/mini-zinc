use std::collections::HashMap;

use ndarray::ArrayD;

use crate::{cgraph::graph::ComputationGraph, op::dtype::TensorValue};

#[test]
pub fn resnet() {
    // let resnet_param = "export/test_resnet/test_resnet.pnnx.param";
    // let resnet_weight = "export/test_resnet/test_resnet.pnnx.bin";
    let bert_param = "export/resnet/resnet18.pnnx.param";
    let bert_weight = "export/resnet/resnet18.pnnx.bin";

    let resnet = ComputationGraph::from_pnnx(bert_param, bert_weight).unwrap();
    // randomize an input of (1,3,224,224)
    let input = HashMap::from([(
        0 as usize,
        TensorValue::Float32(
            ArrayD::from_shape_vec(vec![1, 3, 224, 224], vec![0.0; 1 * 3 * 224 * 224]).unwrap(),
        ),
    )]);
    let result = resnet.compute(&input).unwrap();
    for (k, v) in result {
        eprintln!("{}: {:#?}", k, v);
    }
}

#[test]
pub fn bert() {
    // let resnet_param = "export/test_resnet/test_resnet.pnnx.param";
    // let resnet_weight = "export/test_resnet/test_resnet.pnnx.bin";
    let resnet_param = "export/bert/distilbert-base-uncased-base-fillmask.pnnx.param";
    let resnet_weight = "export/bert/distilbert-base-uncased-base-fillmask.pnnx.bin";

    let resnet = ComputationGraph::from_pnnx(resnet_param, resnet_weight).unwrap();
    let mut input_tokens = vec![0; 1 * 32];
    // 101, 1996, 3007, 1997, 2605, 2003, 1026, 7308, 1028, 1012,  102
    let actual_tokens = vec![
        101, 1996, 3007, 1997, 2605, 2003, 1026, 7308, 1028, 1012, 102,
    ];
    let mut attention_mask = vec![0; 1 * 32];
    for (i, v) in actual_tokens.iter().enumerate() {
        input_tokens[i] = *v;
        attention_mask[i] = 1;
    }
    let input = HashMap::from([
        (
            0 as usize,
            TensorValue::Int64(ArrayD::from_shape_vec(vec![1, 32], input_tokens).unwrap()),
        ),
        (
            1 as usize,
            TensorValue::Int64(ArrayD::from_shape_vec(vec![1, 32], attention_mask).unwrap()),
        ),
    ]);
    let result = resnet.compute(&input).unwrap();

    for (k, v) in result {
        eprintln!("{}", k);
        if let TensorValue::Float32(v) = v {
            eprintln!("{:#?}", v.shape());
            eprintln!("{:#?}", v.as_slice().unwrap()[0]);
        }
    }
}

#[test]
pub fn simple_bert() {
    // let resnet_param = "export/test_resnet/test_resnet.pnnx.param";
    // let resnet_weight = "export/test_resnet/test_resnet.pnnx.bin";
    let resnet_param = "export/test_bert/test_bert.pnnx.param";
    let resnet_weight = "export/test_bert/test_bert.pnnx.bin";

    let resnet = ComputationGraph::from_pnnx(resnet_param, resnet_weight).unwrap();
    let input_tokens = vec![1, 0, 1, 2];
    let input = HashMap::from([
        (
            0 as usize,
            TensorValue::Int64(ArrayD::from_shape_vec(vec![1, 4], input_tokens).unwrap()),
        ),
        (
            1 as usize,
            TensorValue::Int64(ArrayD::from_shape_vec(vec![1, 4], vec![1, 0, 1, 1]).unwrap()),
        ),
    ]);
    let result = resnet.compute(&input).unwrap();

    for (k, v) in result {
        eprintln!("{}", k);
        if let TensorValue::Float32(v) = v {
            eprintln!("{:#?}", v.shape());
            eprintln!("{:#?}", v.as_slice().unwrap()[0]);
        }
    }
}
