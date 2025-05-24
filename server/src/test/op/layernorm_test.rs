use crate::op::conf::{LayerNormConf, ToLayer};
use crate::op::dtype::TensorValue;
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_layernorm_forward() {
    // 创建输入张量：2个样本，每个样本4维特征
    let input = ArrayD::from_shape_vec(
        IxDyn(&[2, 4]),
        vec![
            1.0, 2.0, 3.0, 4.0, // 第一个样本
            5.0, 6.0, 7.0, 8.0, // 第二个样本
        ],
    )
    .unwrap();

    // gamma 和 beta 初始化为常量（1 和 0），即标准化输出不变形
    let gamma = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 0.0, 0.0]).unwrap();

    let conf = LayerNormConf {
        normalized_shape: vec![4], // 对每个样本的特征维度做归一化
        eps: 1e-5,
        elementwise_affine: true,
        weight: TensorValue::Float32(gamma),
        bias: TensorValue::Float32(beta),
    };

    let mut layer = conf.to_layer().unwrap();

    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(out) = &output[0] {
        assert_eq!(out.shape(), &[2, 4]);

        // 手工计算标准化输出
        // 第一个样本: [1,2,3,4] -> mean=2.5, var=1.25
        let expected_sample0 = vec![
            (1.0 - 2.5) / (1.25f32 + 1e-5).sqrt(),
            (2.0 - 2.5) / (1.25f32 + 1e-5).sqrt(),
            (3.0 - 2.5) / (1.25f32 + 1e-5).sqrt(),
            (4.0 - 2.5) / (1.25f32 + 1e-5).sqrt(),
        ];

        // 第二个样本: [5,6,7,8] -> mean=6.5, var=1.25
        let expected_sample1 = vec![
            (5.0 - 6.5) / (1.25f32 + 1e-5).sqrt(),
            (6.0 - 6.5) / (1.25f32 + 1e-5).sqrt(),
            (7.0 - 6.5) / (1.25f32 + 1e-5).sqrt(),
            (8.0 - 6.5) / (1.25f32 + 1e-5).sqrt(),
        ];

        for i in 0..4 {
            assert_abs_diff_eq!(out[[0, i]], expected_sample0[i], epsilon = 1e-4);
            assert_abs_diff_eq!(out[[1, i]], expected_sample1[i], epsilon = 1e-4);
        }
    } else {
        panic!("Expected Float32 tensor");
    }
}
