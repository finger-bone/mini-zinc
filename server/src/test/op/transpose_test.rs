use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};

use crate::op::{
    conf::{ToLayer, TransposeConf},
    layer::TensorValue,
};

#[test]
fn test_transpose_forward() {
    // 创建一个简单的2D输入张量
    let input = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // 创建Transpose层，交换维度0和1
    let transpose_conf = TransposeConf { dim0: 0, dim1: 1 };
    let layer = transpose_conf.to_layer().unwrap();

    // 执行前向传播
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    // 验证输出
    if let TensorValue::Float32(output) = &output[0] {
        // 验证输出形状，应该从[2,3]变为[3,2]
        assert_eq!(output.shape(), &[3, 2]);

        // 验证转置后的值
        assert_abs_diff_eq!(output[[0, 0]], 1.0);
        assert_abs_diff_eq!(output[[0, 1]], 4.0);
        assert_abs_diff_eq!(output[[1, 0]], 2.0);
        assert_abs_diff_eq!(output[[1, 1]], 5.0);
        assert_abs_diff_eq!(output[[2, 0]], 3.0);
        assert_abs_diff_eq!(output[[2, 1]], 6.0);
    } else {
        panic!("Expected Float32 tensor");
    }

    // 测试3D张量的转置
    let input = ArrayD::from_shape_vec(
        IxDyn(&[2, 3, 2]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    )
    .unwrap();

    // 创建Transpose层，交换维度0和1
    let transpose_conf = TransposeConf { dim0: 0, dim1: 1 };
    let layer = transpose_conf.to_layer().unwrap();

    // 执行前向传播
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    // 验证输出
    if let TensorValue::Float32(output) = &output[0] {
        // 验证输出形状，应该从[2,3,2]变为[3,2,2]
        assert_eq!(output.shape(), &[3, 2, 2]);

        // 验证部分转置后的值
        assert_abs_diff_eq!(output[[0, 0, 0]], 1.0);
        assert_abs_diff_eq!(output[[0, 1, 0]], 7.0);
        assert_abs_diff_eq!(output[[1, 0, 0]], 3.0);
        assert_abs_diff_eq!(output[[1, 1, 0]], 9.0);
    } else {
        panic!("Expected Float32 tensor");
    }
}

#[test]
#[should_panic(expected = "Input tensor must have at least 2 dimensions for transpose")]
fn test_transpose_insufficient_dimensions() {
    // 创建一个1D输入张量
    let input = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();

    // 创建Transpose层
    let transpose_conf = TransposeConf { dim0: 0, dim1: 1 };
    let layer = transpose_conf.to_layer().unwrap();

    // 执行前向传播，应该会panic
    layer.forward(&vec![TensorValue::Float32(input)]).unwrap();
}

#[test]
#[should_panic(expected = "Transpose dimensions out of range")]
fn test_transpose_dimensions_out_of_range() {
    // 创建一个2D输入张量
    let input = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // 创建Transpose层，使用超出范围的维度
    let transpose_conf = TransposeConf { dim0: 0, dim1: 2 };
    let layer = transpose_conf.to_layer().unwrap();

    // 执行前向传播，应该会panic
    layer.forward(&vec![TensorValue::Float32(input)]).unwrap();
}