use crate::op::{
    conf::{EmbeddingConf, ToLayer},
    dtype::TensorValue,
};
use ndarray::{ArrayD, array};

#[test]
fn test_embedding_forward_normal_case() {
    // 创建权重张量（4个词，每个嵌入维度3）
    let weight = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![4, 3],
            vec![
                // idx 0
                1.0, 2.0, 3.0, // idx 1
                4.0, 5.0, 6.0, // idx 2
                7.0, 8.0, 9.0, // idx 3
                10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
    );

    // 创建Embedding层配置
    let embedding = EmbeddingConf {
        weight: weight.clone(),
    };
    let mut layer = embedding.to_layer().unwrap();

    // 输入：每个位置是词索引
    let indices = TensorValue::Int64(ArrayD::from_shape_vec(vec![1, 3], vec![0i64, 2, 3]).unwrap());

    // 执行前向传播
    let output = layer.forward(&vec![indices]).unwrap();

    // 验证输出
    if let TensorValue::Float32(output_arr) = &output[0] {
        // ✅ 1. 验证形状
        assert_eq!(output_arr.shape(), &[1, 3, 3]);

        // ✅ 2. 验证每个词向量是否与weight一致
        let expected = array![[
            [1.0, 2.0, 3.0],    // index 0
            [7.0, 8.0, 9.0],    // index 2
            [10.0, 11.0, 12.0]  // index 3
        ]];

        // ✅ 3. 遍历所有值进行精确比较
        for b in 0..1 {
            for i in 0..3 {
                for j in 0..3 {
                    let actual = output_arr[[b, i, j]];
                    let expected_val = expected[[b, i, j]];
                    assert!(
                        (actual - expected_val).abs() < 1e-6,
                        "Mismatch at [{b},{i},{j}]: expected {expected_val}, got {actual}"
                    );
                }
            }
        }
    } else {
        panic!("Expected Float32 output tensor");
    }
}
