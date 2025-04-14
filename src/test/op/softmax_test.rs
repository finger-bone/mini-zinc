// 增强Softmax测试覆盖
use crate::op::{
    conf::{SoftmaxConf, ToLayer},
    layer::{Forward, TensorValue},
};
use ndarray::ArrayD;

#[test]
fn test_softmax_2d() {
    let input = ArrayD::from_shape_vec(
        vec![1, 3], // batch=1, features=3
        vec![1.0, 2.0, 3.0],
    )
    .unwrap();

    let softmax_conf = SoftmaxConf { axis: -1 };
    let layer = softmax_conf.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((output[[0, 2]] > output[[0, 1]]).unwrap());
    }
}

#[test]
fn test_softmax_4d() {
    let input = ArrayD::from_shape_vec(
        vec![1, 2, 2, 2], // batch=1, H=2, W=2, channels=2
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let softmax_conf = SoftmaxConf { axis: 1 };
    let layer = softmax_conf.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        // 沿axis=1（高度维度）的和应为1
        for c in 0..2 {
            for w in 0..2 {
                let sum: f32 = (0..2).map(|h| output[[0, h, w, c]]).sum();
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
    }
}