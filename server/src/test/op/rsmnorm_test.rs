use crate::op::conf::{RSMNormConf, ToLayer};
use crate::op::dtype::TensorValue;
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_rsmnorm_forward() {
    // 创建输入张量：2个样本，每个样本4维特征
    let input =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();

    // gamma 初始化为常量 1（即不缩放）
    let gamma = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

    let conf = RSMNormConf {
        normalized_shape: vec![4],
        eps: 1e-5,
        elementwise_affine: true,
        weight: TensorValue::Float32(gamma),
    };

    let mut layer = conf.to_layer().unwrap();

    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    if let TensorValue::Float32(out) = &output[0] {
        assert_eq!(out.shape(), &[2, 4]);

        // 计算 sample0 的 RMSNorm
        let expected_sample0 = {
            let vals = vec![1.0, 2.0, 3.0, 4.0];
            let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32 + 1e-5).sqrt();
            vals.iter().map(|&v| v / rms).collect::<Vec<f32>>()
        };

        let expected_sample1 = {
            let vals = vec![5.0, 6.0, 7.0, 8.0];
            let rms = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32 + 1e-5).sqrt();
            vals.iter().map(|&v| v / rms).collect::<Vec<f32>>()
        };

        for i in 0..4 {
            assert_abs_diff_eq!(out[[0, i]], expected_sample0[i], epsilon = 1e-4);
            assert_abs_diff_eq!(out[[1, i]], expected_sample1[i], epsilon = 1e-4);
        }
    } else {
        panic!("Expected Float32 tensor");
    }
}
