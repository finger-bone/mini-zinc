use ndarray::ArrayD;

use crate::op::conf::{MaskedFillConf, ToLayer};
use crate::op::dtype::TensorValue;

// Add to masked_fill_test.rs
#[test]
fn test_masked_fill_all_true() {
    let conf = MaskedFillConf { value: 5.0 };
    let mut layer = conf.to_layer().unwrap();

    let data = ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let mask = ArrayD::from_shape_vec(vec![3], vec![true; 3]).unwrap();

    let output = layer
        .forward(&vec![
            TensorValue::Float32(data),
            TensorValue::Boolean(mask),
        ])
        .unwrap();

    if let TensorValue::Float32(result) = &output[0] {
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.as_slice().unwrap(), &[5.0, 5.0, 5.0]);
    } else {
        panic!("Unexpected tensor type");
    }
}

#[test]
fn test_masked_fill_all_false() {
    let conf = MaskedFillConf { value: 10.0 };
    let mut layer = conf.to_layer().unwrap();

    let data = ArrayD::from_shape_vec(vec![4], vec![1.5, 2.5, 3.5, 4.5]).unwrap();
    let mask = ArrayD::from_shape_vec(vec![4], vec![false; 4]).unwrap();

    let output = layer
        .forward(&vec![
            TensorValue::Float32(data.clone()),
            TensorValue::Boolean(mask),
        ])
        .unwrap();

    if let TensorValue::Float32(result) = &output[0] {
        assert_eq!(result.shape(), &[4]);
        assert_eq!(result, &data);
    } else {
        panic!("Unexpected tensor type");
    }
}

#[test]
fn test_masked_fill_2d() {
    let conf = MaskedFillConf { value: -1.0 };
    let mut layer = conf.to_layer().unwrap();

    let data = ArrayD::from_shape_vec(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    let mask = ArrayD::from_shape_vec(vec![2, 2], vec![false, true, true, false]).unwrap();

    let expected = ArrayD::from_shape_vec(vec![2, 2], vec![10.0, -1.0, -1.0, 40.0]).unwrap();

    let output = layer
        .forward(&vec![
            TensorValue::Float32(data),
            TensorValue::Boolean(mask),
        ])
        .unwrap();

    if let TensorValue::Float32(result) = &output[0] {
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(*result, expected);
    } else {
        panic!("Unexpected tensor type");
    }
}
