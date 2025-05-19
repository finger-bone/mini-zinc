use approx::assert_abs_diff_eq;
use ndarray::{arr3, ArrayD, IxDyn};

use crate::op::conf::{ToLayer, TransposeConf};
use crate::op::dtype::TensorValue;

#[test]
fn test_transpose_forward() {
    // ---------- 测试 2D 转置 ----------
    let input = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let transpose_conf = TransposeConf { dim0: 0, dim1: 1 };
    let layer = transpose_conf.to_layer().unwrap();

    let output = layer.forward(&vec![TensorValue::Float32(input.clone())]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[3, 2]);
        let expected = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![
            1.0, 4.0,
            2.0, 5.0,
            3.0, 6.0,
        ]).unwrap();

        for ((_, &actual), &expected_val) in output.indexed_iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected_val, epsilon = 1e-6);
        }
    } else {
        panic!("Expected Float32 tensor");
    }

    // ---------- 测试 3D 转置 ----------
    let input = ArrayD::from_shape_vec(
        IxDyn(&[2, 3, 2]),
        vec![
            1.0, 2.0,  // (0,0)
            3.0, 4.0,  // (0,1)
            5.0, 6.0,  // (0,2)
            7.0, 8.0,  // (1,0)
            9.0, 10.0, // (1,1)
            11.0, 12.0 // (1,2)
        ],
    ).unwrap();

    let output = layer.forward(&vec![TensorValue::Float32(input.clone())]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[3, 2, 2]);

        let expected = arr3(&[
            [[1.0, 2.0], [7.0, 8.0]],
            [[3.0, 4.0], [9.0, 10.0]],
            [[5.0, 6.0], [11.0, 12.0]],
        ])
        .into_dyn();

        for ((_, &actual), &expected_val) in output.indexed_iter().zip(expected.iter()) {
            assert_abs_diff_eq!(actual, expected_val, epsilon = 1e-6);
        }
    } else {
        panic!("Expected Float32 tensor");
    }
}

#[test]
#[should_panic(expected = "Input tensor must have at least 2 dimensions for transpose")]
fn test_transpose_insufficient_dimensions() {
    let input = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let transpose_conf = TransposeConf { dim0: 0, dim1: 1 };
    let layer = transpose_conf.to_layer().unwrap();
    layer.forward(&vec![TensorValue::Float32(input)]).unwrap();
}

#[test]
#[should_panic(expected = "Transpose dimensions out of range")]
fn test_transpose_dimensions_out_of_range() {
    let input = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let transpose_conf = TransposeConf { dim0: 0, dim1: 2 };
    let layer = transpose_conf.to_layer().unwrap();
    layer.forward(&vec![TensorValue::Float32(input)]).unwrap();
}