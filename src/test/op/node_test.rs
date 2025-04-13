use crate::cgraph::{
    node::{CGNode, CGNodeOp},
    pnnx_reader::{PNNXKV, PNNXKVType, PNNXLine},
};
use crate::op::{
    conf::{FlattenConf, ToLayer},
    layer::TensorValue,
};
use std::collections::HashMap;

#[test]
fn test_cgnode_from_flatten_line() {
    // Create a mock PNNXLine for torch.flatten
    let mut kvs = Vec::new();
    kvs.push(PNNXKV {
        kv_type: PNNXKVType::Attr,
        key: "start_dim".to_string(),
        value: "1".to_string(),
    });
    kvs.push(PNNXKV {
        kv_type: PNNXKVType::Attr,
        key: "end_dim".to_string(),
        value: "2".to_string(),
    });

    let line = PNNXLine {
        op_type: "torch.flatten".to_string(),
        op_name: "flatten_0".to_string(),
        input_idx_list: vec![0],
        output_idx_list: vec![1],
        kvs,
    };

    let weights = HashMap::new(); // No weights needed for Flatten

    let node = CGNode::from_line(&line, &weights).unwrap();

    // Validate the parsed node
    assert_eq!(node.name, "flatten_0");
    match &node.op {
        CGNodeOp::Op(layer) => {
            let input_tensor = TensorValue::Float32(ndarray::ArrayD::zeros(vec![2, 3, 4])); // 3-dimensional tensor
            if let TensorValue::Float32(output_tensor) =
                &layer.forward(&vec![input_tensor]).unwrap()[0]
            {
                // Ensure forward pass works (basic validation)
                assert_eq!(output_tensor.shape(), &[2, 12]); // Flattened shape should be [2, 12]
            } else {
                panic!("Unexpected tensor type");
            }
        }
        _ => panic!("Expected Op variant"),
    }
}
