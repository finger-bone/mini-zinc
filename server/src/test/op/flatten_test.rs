use crate::op::{
    conf::{FlattenConf, ToLayer},
    layer::TensorValue,
};
use ndarray::ArrayD;

#[test]
fn test_flatten_forward() {
    // Test flattening a 3D tensor into a 2D tensor
    let flatten = FlattenConf {
        start_dim: 1,
        end_dim: 2,
    };
    let layer = flatten.to_layer().unwrap();

    let input =
        ArrayD::from_shape_vec(vec![2, 3, 4], (1..=24).map(|x| x as f32).collect()).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    let expected =
        ArrayD::from_shape_vec(vec![2, 12], (1..=24).map(|x| x as f32).collect()).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output, expected);
    } else {
        panic!("Output is not Float32");
    }
}

#[test]
fn test_flatten_with_negative_dims() {
    // Test flattening with negative start_dim and end_dim
    let flatten = FlattenConf {
        start_dim: -2,
        end_dim: -1,
    };
    let layer = flatten.to_layer().unwrap();

    let input =
        ArrayD::from_shape_vec(vec![2, 3, 4], (1..=24).map(|x| x as f32).collect()).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    let expected =
        ArrayD::from_shape_vec(vec![2, 12], (1..=24).map(|x| x as f32).collect()).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output, expected);
    } else {
        panic!("Output is not Float32");
    }
}

#[test]
#[should_panic(expected = "Invalid start_dim or end_dim for Flatten")]
fn test_flatten_invalid_dims() {
    // Test invalid dimensions configuration
    let flatten = FlattenConf {
        start_dim: 2,
        end_dim: 1, // Invalid: start_dim > end_dim
    };
    let layer = flatten.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 3, 4], vec![1.0; 24]).unwrap();
    layer.forward(&vec![TensorValue::Float32(input)]).unwrap(); // This should panic
}
