use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};

use crate::op::conf::{SiLUConf, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_silu_forward() {
    // 创建一个简单的输入张量
    let input = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, 0.0, 2.0]).unwrap();

    // 创建SiLU层
    let silu_conf = SiLUConf {};
    let mut layer = silu_conf.to_layer().unwrap();

    // 执行前向传播
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    // 验证输出
    if let TensorValue::Float32(output) = &output[0] {
        // SiLU(-2.0) ≈ -0.2384
        assert_abs_diff_eq!(output[0], -0.2384, epsilon = 0.01);
        // SiLU(0.0) = 0.0
        assert_abs_diff_eq!(output[1], 0.0, epsilon = 0.01);
        // SiLU(2.0) ≈ 1.7616
        assert_abs_diff_eq!(output[2], 1.7616, epsilon = 0.01);
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

        // SiLU(-1.0) ≈ -0.2689
        assert_abs_diff_eq!(output[[0, 0]], -0.2689, epsilon = 0.01);
        // SiLU(-0.5) ≈ -0.1888
        assert_abs_diff_eq!(output[[0, 1]], -0.1888, epsilon = 0.01);
        // SiLU(0.5) ≈ 0.3112
        assert_abs_diff_eq!(output[[1, 0]], 0.3112, epsilon = 0.01);
        // SiLU(1.0) ≈ 0.7311
        assert_abs_diff_eq!(output[[1, 1]], 0.7311, epsilon = 0.01);
    } else {
        panic!("Expected Float32 tensor");
    }
}