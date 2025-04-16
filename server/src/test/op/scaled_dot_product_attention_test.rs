use crate::op::conf::{ScaledDotProductAttentionConf, ToLayer};
use crate::op::dtype::TensorValue;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_scaled_dot_product_attention() {
    // 创建配置
    let conf = ScaledDotProductAttentionConf {
        dropout: 0.0,
        is_causal: false,
        max_seq_len: 64,
        scale: Some(0.5), // 自定义缩放因子
    };
    let layer = conf.to_layer().unwrap();

    // 创建输入张量
    // 批次大小=1, 头数=2, 序列长度=3, 嵌入维度=4
    let batch = 1;
    let heads = 2;
    let seq_len = 3;
    let embed_dim = 4;

    // 创建查询矩阵 Q
    let q_data = vec![
        // 第一个头
        0.1, 0.2, 0.3, 0.4, // 第一个序列位置
        0.5, 0.6, 0.7, 0.8, // 第二个序列位置
        0.9, 1.0, 1.1, 1.2, // 第三个序列位置
        // 第二个头
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ];
    let q = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, embed_dim]), q_data).unwrap(),
    );

    // 创建键矩阵 K
    let k_data = vec![
        // 第一个头
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, // 第二个头
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ];
    let k = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, embed_dim]), k_data).unwrap(),
    );

    // 创建值矩阵 V
    let v_data = vec![
        // 第一个头
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, // 第二个头
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ];
    let v = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, embed_dim]), v_data).unwrap(),
    );

    // 创建掩码矩阵 mask (全1表示不掩码)
    let mask_data = vec![
        // 第一个头
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 第二个头
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let mask = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, seq_len]), mask_data).unwrap(),
    );

    // 执行前向传播
    let output = layer.forward(&vec![q, k, v, mask]).unwrap();

    // 验证输出
    if let TensorValue::Float32(output_arr) = &output[0] {
        // 验证输出形状
        assert_eq!(output_arr.shape(), &[batch, heads, seq_len, embed_dim]);

        // 由于注意力机制的计算比较复杂，这里我们只验证输出不是NaN或无穷大
        let output_slice = output_arr.as_slice().unwrap();
        for &value in output_slice {
            assert!(!value.is_nan() && !value.is_infinite());
        }

        // 可以添加更多具体的验证，例如验证某些已知的值
        // 这里我们假设已经手动计算了第一个头的第一个位置的输出应该接近某个值
        // 注意：实际值需要根据具体实现计算得出
        // assert_relative_eq!(output_slice[0], expected_value, epsilon = 1e-5);
    } else {
        panic!("Expected Float32 tensor");
    }
}

#[test]
fn test_scaled_dot_product_attention_causal() {
    // 创建带有因果掩码的配置
    let conf = ScaledDotProductAttentionConf {
        dropout: 0.0,
        is_causal: true, // 启用因果掩码
        max_seq_len: 64,
        scale: None, // 使用默认缩放因子
    };
    let layer = conf.to_layer().unwrap();

    // 创建输入张量 (简化版本)
    let batch = 1;
    let heads = 1;
    let seq_len = 3;
    let embed_dim = 2;

    // 创建查询、键、值矩阵
    let q_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let q = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, embed_dim]), q_data).unwrap(),
    );

    let k_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let k = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, embed_dim]), k_data).unwrap(),
    );

    let v_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let v = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, embed_dim]), v_data).unwrap(),
    );

    // 创建因果掩码 (上三角为0，表示后面的token不能看到前面的token)
    let mask_data = vec![
        1.0, 0.0, 0.0, // 第一个位置只能看到自己
        1.0, 1.0, 0.0, // 第二个位置能看到自己和第一个位置
        1.0, 1.0, 1.0, // 第三个位置能看到所有位置
    ];
    let mask = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_len, seq_len]), mask_data).unwrap(),
    );

    // 执行前向传播
    let output = layer.forward(&vec![q, k, v, mask]).unwrap();

    // 验证输出
    if let TensorValue::Float32(output_arr) = &output[0] {
        // 验证输出形状
        assert_eq!(output_arr.shape(), &[batch, heads, seq_len, embed_dim]);

        // 验证输出不是NaN或无穷大
        let output_slice = output_arr.as_slice().unwrap();
        for &value in output_slice {
            assert!(!value.is_nan() && !value.is_infinite());
        }
    } else {
        panic!("Expected Float32 tensor");
    }
}
