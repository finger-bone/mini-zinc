use ndarray::ArrayD;

use crate::op::conf::{Pool2dConf, PoolType, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_maxpool_forward() {
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

    // Create a 2x2 max pooling layer
    let pool = Pool2dConf {
        kernel_size: vec![2, 2],
        stride: vec![2, 2],
        padding: vec![0, 0],
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
fn test_avgpool_forward() {
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

    // Create a 2x2 average pooling layer
    let pool = Pool2dConf {
        kernel_size: vec![2, 2],
        stride: vec![2, 2],
        padding: vec![0, 0],
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
fn test_maxpool_value() {
    // 1x1x3x3 输入
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(
            vec![1, 1, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
    );

    // MaxPool 2x2, stride=1, no padding
    let pool = Pool2dConf {
        kernel_size: vec![2, 2],
        stride: vec![1, 1],
        padding: vec![0, 0],
        pool_type: PoolType::Max,
    };

    let mut layer = pool.to_layer().unwrap();
    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 1, 2, 2]);

        // Max over windows:
        // [[1,2],[4,5]] => 5
        // [[2,3],[5,6]] => 6
        // [[4,5],[7,8]] => 8
        // [[5,6],[8,9]] => 9
        let expected: ndarray::ArrayBase<
            ndarray::OwnedRepr<f32>,
            ndarray::Dim<ndarray::IxDynImpl>,
        > = ArrayD::from_shape_vec(vec![1, 1, 2, 2], vec![5.0, 6.0, 8.0, 9.0]).unwrap();

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

// 添加多通道max pooling测试
#[test]
fn test_maxpool_multi_channel() {
    // 创建2通道输入
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

    let pool = Pool2dConf {
        kernel_size: vec![2, 2],
        stride: vec![2, 2],
        padding: vec![0, 0],
        pool_type: PoolType::Max,
    };

    let mut layer = pool.to_layer().unwrap();
    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Float32(output_array) = &output[0] {
        assert_eq!(output_array.shape(), &[1, 2, 2, 2]);

        let expected = ArrayD::from_shape_vec(
            vec![1, 2, 2, 2],
            vec![
                // 通道1
                6.0, 8.0, 14.0, 16.0, // 通道2
                7.0, 9.0, 15.0, 17.0,
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
