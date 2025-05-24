use ndarray::ArrayD;

use crate::op::conf::{ReLUConf, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_relu_forward() {
    let relu = ReLUConf { threshold: 0.0 };
    let mut layer = relu.to_layer().unwrap();

    // 测试正数保持不变
    let input = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    // assert_eq!(output[0], input);
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output, input);
    } else {
        panic!("Expected Float32 output");
    }

    // 测试负数变为0
    let input = ArrayD::from_shape_vec(vec![2, 2], vec![-1.0, -2.0, 0.0, 1.0]).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(
            output,
            ArrayD::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 0.0, 1.0]).unwrap()
        );
    } else {
        panic!("Expected Float32 output");
    }

    // 测试自定义阈值
    let relu = ReLUConf { threshold: 2.0 };
    let mut layer = relu.to_layer().unwrap();
    let input = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    let expected = ArrayD::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 3.0, 4.0]).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output, expected);
    } else {
        panic!("Expected Float32 output");
    }
}
