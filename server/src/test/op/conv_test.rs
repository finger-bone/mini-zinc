// 新增Conv2d测试文件
use crate::op::{
    conf::{Conv2dConf, ToLayer},
    layer::{Forward, TensorValue},
};
use ndarray::ArrayD;

#[test]
fn test_conv2d_basic() {
    // 创建简单输入和卷积核
    let input = ArrayD::from_shape_vec(
        vec![1, 2, 2, 1], // batch=1, H=2, W=2, in_channels=1
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    let conv_conf = Conv2dConf {
        in_channels: 1,
        out_channels: 1,
        kernel_size: vec![2, 2],
        stride: vec![1, 1],
        padding: vec![0, 0],
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(
                vec![1, 1, 2, 2], // out_channels, in_channels, kernel_H, kernel_W
                vec![0.1, 0.2, 0.3, 0.4],
            )
            .unwrap(),
        ),
        bias: TensorValue::Float32(ArrayD::from_shape_vec(vec![1], vec![0.0]).unwrap()),
    };

    let mut layer = conv_conf.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    // 预期输出形状应为 [1, 1, 1, 1]（因为输入2x2 kernel 2x2 stride 1）
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[1, 1, 1, 1]);
        assert!((output[[0, 0, 0, 0]] - 2.5).abs() < 1e-6);
    }
}

#[test]
#[should_panic(expected = "Input channels mismatch")]
fn test_conv2d_channel_mismatch() {
    let conv_conf = Conv2dConf {
        in_channels: 3,
        out_channels: 1,
        kernel_size: vec![3, 3],
        weights: TensorValue::Float32(ArrayD::zeros(vec![1, 2, 3, 3])), // 输入通道数不匹配
        ..Default::default()
    };
    conv_conf.to_layer().unwrap();
}