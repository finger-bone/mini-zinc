// server/src/test/op/embedding_test.rs
use crate::op::{
    conf::{EmbeddingConf, ToLayer},
    dtype::TensorValue,
};
use ndarray::ArrayD;

#[test]
fn test_embedding_forward_normal_case() {
    // 创建权重张量（5个词，每个维度3）
    let weight = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![4, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
    );

    // 创建Embedding层配置
    let embedding = EmbeddingConf {
        weight: weight.clone(),
    };
    let layer = embedding.to_layer().unwrap();

    let indices = TensorValue::Int64(
        ArrayD::from_shape_vec(
            vec![1, 3], // New shape: 1D array with 3 elements
            vec![0i64, 2, 3],
        )
        .unwrap(),
    );
    // 执行前向传播
    let output = layer.forward(&vec![indices]).unwrap();

    // 验证输出形状和内容
    if let TensorValue::Float32(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[1, 3, 3]);
        assert_eq!(output_arr[[0, 0, 0]], 1.0); // 第0个词的第一个维度
        assert_eq!(output_arr[[0, 1, 2]], 9.0); // 第2个词的第三个维度
        assert_eq!(output_arr[[0, 2, 1]], 11.0); // 第3个词的第二个维度（索引4对应第五行）
    }
}
