use ndarray::ArrayD;

use crate::op::{
    conf::{Conv2dConf, ToLayer},
    layer::{Forward, TensorValue},
};

#[test]
fn test_conv_forward() {
    // Create a simple 1x1x4x4 input (batch_size=1, channels=1, height=4, width=4)
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 1, 4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );

    // Create a basic 3x3 convolution with 2 output channels
    let conv = Conv2dConf {
        kernel_size: vec![3, 3],
        stride: vec![1, 1],
        padding: vec![1, 1],
        filters: 2,
        dilation: vec![1, 1],
        groups: 1,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![2, 1, 3, 3], vec![0.1; 18]).unwrap(),
        ),
        bias: TensorValue::Float32(ArrayD::from_shape_vec(vec![2], vec![0.1; 2]).unwrap()),
    };

    let layer = conv.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x2x4x4 with padding)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 2, 4, 4]);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_conv_stride() {
    // Create a simple 1x1x4x4 input (batch_size=1, channels=1, height=4, width=4)
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 1, 4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );

    // Create a strided convolution
    let conv = Conv2dConf {
        kernel_size: vec![3, 3],
        stride: vec![2, 2],
        padding: vec![1, 1],
        filters: 2,
        dilation: vec![1, 1],
        groups: 1,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![2, 1, 3, 3], vec![0.1; 18]).unwrap(),
        ),
        bias: TensorValue::Float32(ArrayD::from_shape_vec(vec![2], vec![0.1; 2]).unwrap()),
    };

    let layer = conv.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x2x2x2 with stride=2)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 2, 2, 2]);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_conv_dilation() {
    // Create a simple 1x1x6x6 input
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 1, 6, 6],
            (1..37).map(|x| x as f32).collect::<Vec<f32>>(),
        )
        .unwrap(),
    );

    // Create a dilated convolution
    let conv = Conv2dConf {
        kernel_size: vec![3, 3],
        stride: vec![1, 1],
        padding: vec![2, 2],
        filters: 1,
        dilation: vec![2, 2],
        groups: 1,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![1, 1, 3, 3], vec![0.1; 9]).unwrap(),
        ),
        bias: TensorValue::Float32(ArrayD::from_shape_vec(vec![1], vec![0.1]).unwrap()),
    };

    let layer = conv.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x1x6x6 with dilation=2 and appropriate padding)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 1, 6, 6]);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_conv_groups() {
    // Create input with 4 input channels
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 4, 4, 4],
            (1..65).map(|x| x as f32).collect::<Vec<f32>>(),
        )
        .unwrap(),
    );

    // Create grouped convolution (2 groups)
    let conv = Conv2dConf {
        kernel_size: vec![3, 3],
        stride: vec![1, 1],
        padding: vec![1, 1],
        filters: 4, // Must be divisible by groups
        dilation: vec![1, 1],
        groups: 2,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![4, 2, 3, 3], vec![0.1; 72]).unwrap(),
        ),
        bias: TensorValue::Float32(ArrayD::from_shape_vec(vec![4], vec![0.1; 4]).unwrap()),
    };

    let layer = conv.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x4x4x4)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 4, 4, 4]);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_conv_no_padding() {
    // Create a simple 1x1x5x5 input
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 1, 5, 5],
            (1..26).map(|x| x as f32).collect::<Vec<f32>>(),
        )
        .unwrap(),
    );

    // Create convolution without padding
    let conv = Conv2dConf {
        kernel_size: vec![3, 3],
        stride: vec![1, 1],
        padding: vec![0, 0],
        filters: 2,
        dilation: vec![1, 1],
        groups: 1,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![2, 1, 3, 3], vec![0.1; 18]).unwrap(),
        ),
        bias: TensorValue::Float32(ArrayD::from_shape_vec(vec![2], vec![0.1; 2]).unwrap()),
    };

    let layer = conv.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x2x3x3 without padding)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 2, 3, 3]);
    } else {
        panic!("Expected Float32 output");
    }
}
