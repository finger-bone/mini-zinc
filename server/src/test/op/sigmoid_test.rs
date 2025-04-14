use crate::op::{
    conf::{SigmoidConf, ToLayer},
    layer::{Forward, TensorValue},
};
use ndarray::ArrayD;

#[test]
fn test_sigmoid_range() {
    let input = ArrayD::from_shape_vec(
        vec![1, 4], // batch=1, features=4
        vec![-100.0, 0.0, 10.0, 1000.0],
    )
    .unwrap();

    let layer = SigmoidConf {}.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert!((output[[0,0]] - 0.0).abs() < 1e-6);
        assert!((output[[0,1]] - 0.5).abs() < 1e-6);
        assert!((output[[0,2]] - 1.0).abs() < 1e-6);
        assert!((output[[0,3]] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_sigmoid_derivative() {
    let input = ArrayD::from_elem(ndarray::IxDyn(&[1,1]), 0.0);
    let layer = SigmoidConf {}.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        let val = output[[0,0]];
        let derivative = val * (1.0 - val);
        assert!((derivative - 0.25).abs() < 1e-6);
    }
}