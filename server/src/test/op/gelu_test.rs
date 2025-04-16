use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};

use crate::op::conf::{GeLUConf, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_gelu_forward() {
    // 创建一个简单的输入张量
    let input = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, 0.0, 2.0]).unwrap();

    // 创建GeLU层
    let gelu_conf = GeLUConf {};
    let layer = gelu_conf.to_layer().unwrap();

    // 执行前向传播
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    // 验证输出
    if let TensorValue::Float32(output) = &output[0] {
        // GeLU(-2.0) ≈ -0.046
        assert_abs_diff_eq!(output[0], -0.046, epsilon = 0.01);
        // GeLU(0.0) = 0.0
        assert_abs_diff_eq!(output[1], 0.0, epsilon = 0.01);
        // GeLU(2.0) ≈ 1.954
        assert_abs_diff_eq!(output[2], 1.954, epsilon = 0.01);
    } else {
        panic!("Expected Float32 tensor");
    }

    // 测试更复杂的输入
    let input = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -0.5, 0.5, 1.0]).unwrap();

    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        // 验证输出形状
        assert_eq!(output.shape(), &[2, 2]);

        // GeLU(-1.0) ≈ -0.159
        assert_abs_diff_eq!(output[[0, 0]], -0.159, epsilon = 0.01);
        // GeLU(-0.5) ≈ -0.154
        assert_abs_diff_eq!(output[[0, 1]], -0.154, epsilon = 0.01);
        // GeLU(0.5) ≈ 0.346
        assert_abs_diff_eq!(output[[1, 0]], 0.346, epsilon = 0.01);
        // GeLU(1.0) ≈ 0.841
        assert_abs_diff_eq!(output[[1, 1]], 0.841, epsilon = 0.01);
    } else {
        panic!("Expected Float32 tensor");
    }
}
