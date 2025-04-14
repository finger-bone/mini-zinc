use crate::op::{
    conf::{LayerNormConf, ToLayer},
    layer::TensorValue,
};
use approx::assert_abs_diff_eq;
use ndarray::{ArrayD, Axis, IxDyn};

#[test]
fn test_layernorm_forward() {
    // 输入形状: [batch=2, seq_len=1, embed_dim=3]
    let input = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 3]),
        vec![
            1.0, 2.0, 3.0, // 样本1
            4.0, 5.0, 6.0,
        ], // 样本2
    )
    .unwrap();

    // 配置：normalized_shape=3，weight=2.0，bias=1.0
    let layer_norm_conf = LayerNormConf {
        normalized_shape: vec![3],
        eps: 1e-12,
        elementwise_affine: true,
        weight: TensorValue::Float32(ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0; 3]).unwrap()),
        bias: TensorValue::Float32(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0; 3]).unwrap()),
    };

    let layer = layer_norm_conf.to_layer().unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();

    if let TensorValue::Float32(output_arr) = &output[0] {
        // 构造期望输出：对最后一维（embed_dim）做归一化后 *2 +1
        let mut expected = input.clone();

        for mut sample in expected.outer_iter_mut() {
            for mut token in sample.outer_iter_mut() {
                let mean = token.mean().unwrap();
                let var = token.var_axis(Axis(0), 0.0);
                let std = var.mapv(|v| (v + 1e-12).sqrt());

                token.zip_mut_with(&std, |x, s| {
                    *x = ((*x - mean) / s) * 2.0 + 1.0;
                });
            }
        }

        for (out, exp) in output_arr.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(out, exp, epsilon = 1e-5);
        }
    } else {
        panic!("Output is not Float32");
    }
}
