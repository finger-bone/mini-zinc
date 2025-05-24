use ndarray::ArrayD;

use crate::op::conf::{AdaptivePool2dConf, PoolType, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_adaptive_maxpool_forward() {
    // Create a simple 1x1x4x4 input (batch_size=1, channels=1, height=4, width=4)
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 1, 4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );

    // Create an adaptive max pooling layer that outputs 2x2
    let pool = AdaptivePool2dConf {
        output_size: vec![2, 2],
        pool_type: PoolType::Max,
    };

    let mut layer = pool.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x1x2x2)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 1, 2, 2]);

        // Check max pooling values
        let expected =
            ArrayD::from_shape_vec(vec![1, 1, 2, 2], vec![6.0, 8.0, 14.0, 16.0]).unwrap();

        // Due to potential floating-point precision issues, we'll check approximately
        for i in 0..output_array.len() {
            assert!(
                (output_array.as_slice().unwrap()[i] - expected.as_slice().unwrap()[i]).abs()
                    < 1e-5
            );
        }
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_adaptive_avgpool_forward() {
    // Create a simple 1x1x4x4 input (batch_size=1, channels=1, height=4, width=4)
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 1, 4, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap(),
    );

    // Create an adaptive average pooling layer that outputs 2x2
    let pool = AdaptivePool2dConf {
        output_size: vec![2, 2],
        pool_type: PoolType::Avg,
    };

    let mut layer = pool.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x1x2x2)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 1, 2, 2]);

        // Check average pooling values
        let expected =
            ArrayD::from_shape_vec(vec![1, 1, 2, 2], vec![3.5, 5.5, 11.5, 13.5]).unwrap();

        // Due to potential floating-point precision issues, we'll check approximately
        for i in 0..output_array.len() {
            assert!(
                (output_array.as_slice().unwrap()[i] - expected.as_slice().unwrap()[i]).abs()
                    < 1e-5
            );
        }
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_adaptive_pool_multi_channel() {
    // Create a 1x2x4x4 input (batch_size=1, channels=2, height=4, width=4)
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 2, 4, 4],
            vec![
                // Channel 1
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, // Channel 2
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0,
            ],
        )
        .unwrap(),
    );

    // Create an adaptive max pooling layer that outputs 2x2
    let pool = AdaptivePool2dConf {
        output_size: vec![2, 2],
        pool_type: PoolType::Max,
    };

    let mut layer = pool.to_layer().unwrap();

    // Forward pass
    let output = layer.forward(&vec![input]).unwrap();

    // Check output shape (should be 1x2x2x2)
    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 2, 2, 2]);

        // Check max pooling values for both channels
        let expected = ArrayD::from_shape_vec(
            vec![1, 2, 2, 2],
            vec![
                // Channel 1
                6.0, 8.0, 14.0, 16.0, // Channel 2
                7.0, 9.0, 15.0, 17.0,
            ],
        )
        .unwrap();

        // Due to potential floating-point precision issues, we'll check approximately
        for i in 0..output_array.len() {
            assert!(
                (output_array.as_slice().unwrap()[i] - expected.as_slice().unwrap()[i]).abs()
                    < 1e-5
            );
        }
    } else {
        panic!("Expected Float32 output");
    }
}

// 添加多通道自适应平均池化测试
#[test]
fn test_adaptive_avgpool_multi_channel() {
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 2, 4, 4],
            vec![
                // 通道1
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, // 通道2
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0,
            ],
        )
        .unwrap(),
    );

    let pool = AdaptivePool2dConf {
        output_size: vec![2, 2],
        pool_type: PoolType::Avg,
    };

    let mut layer = pool.to_layer().unwrap();
    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 2, 2, 2]);

        let expected = ArrayD::from_shape_vec(
            vec![1, 2, 2, 2],
            vec![
                // 通道1
                3.5, 5.5, 11.5, 13.5, // 通道2
                4.5, 6.5, 12.5, 14.5,
            ],
        )
        .unwrap();

        for i in 0..output_array.len() {
            assert!(
                (output_array.as_slice().unwrap()[i] - expected.as_slice().unwrap()[i]).abs()
                    < 1e-5
            );
        }
    }
}
