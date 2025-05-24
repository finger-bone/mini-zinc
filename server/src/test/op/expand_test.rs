// 新增expand层单元测试文件
use crate::op::conf::{ExpandConf, ToLayer};
use crate::op::dtype::TensorValue;
use ndarray::ArrayD;

#[test]
fn test_expand_forward() {
    // 测试可广播的情况
    let expand = ExpandConf {
        shape: vec![2, 3], // 目标形状
    };
    let mut layer = expand.to_layer().unwrap();

    // 输入形状为[2,1]，可以广播到[2,3]
    let input = ArrayD::from_shape_vec(vec![2, 1], vec![1.0, 2.0]).unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[2, 3]);
        assert_eq!(output_arr[[0, 0]], 1.0);
        assert_eq!(output_arr[[1, 2]], 2.0);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_expand_incompatible_shape() {
    // 测试无法广播的情况
    let expand = ExpandConf {
        shape: vec![2, 4], // 与输入形状不兼容
    };
    let mut layer = expand.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let result = layer.forward(&vec![TensorValue::Float32(input)]);

    // 验证错误是否被正确抛出
    assert!(result.is_err());
    assert!(format!("{:?}", result.unwrap_err()).contains("Cannot broadcast"));
}

#[test]
fn test_expand_no_change() {
    // 输入形状与目标形状一致
    let expand = ExpandConf { shape: vec![3, 2] };
    let mut layer = expand.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![3, 2], vec![1.0; 6]).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    if let TensorValue::Float32(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[3, 2]);
        assert_eq!(output_arr, &input);
    }
}
