// 新增MaxPool2d测试
use crate::op::{
    conf::{MaxPool2dConf, ToLayer},
    layer::{Forward, TensorValue},
};
use ndarray::ArrayD;

#[test]
fn test_max_pool2d_basic() {
    let input = ArrayD::from_shape_vec(
        vec![1, 2, 2, 1], // batch=1, H=2, W=2, channels=1
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    let pool_conf = MaxPool2dConf {
        kernel_size: vec![2, 2],
        stride: vec![2, 2],
        padding: vec![0, 0],
    };

    let layer = pool_conf.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[1, 1, 1, 1]);
        assert!((output[[0, 0, 0, 0]] - 4.0).abs() < 1e-6);
    }
}

#[test]
fn test_max_pool2d_padding() {
    let input = ArrayD::from_shape_vec(
        vec![1, 3, 3, 1],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();

    let pool_conf = MaxPool2dConf {
        kernel_size: vec![3, 3],
        stride: vec![1, 1],
        padding: vec![1, 1],
    };

    let layer = pool_conf.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[1, 3, 3, 1]);
        assert!((output[[0, 0, 0, 0]] - 9.0).abs() < 1e-6);
    }
}