use crate::op::conf::{TensorToConf, ToLayer};
use crate::op::dtype::{DataType, TensorValue};
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_float32_to_int64() {
    let conf = TensorToConf {
        target_dtype: DataType::Int64,
    };
    let mut layer = conf.to_layer().unwrap();

    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.5f32, -2.3f32, 4.0f32]).unwrap(),
    );

    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Int64(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[3]);
        assert_eq!(output_arr.as_slice().unwrap(), &[1, -2, 4]);
    } else {
        panic!("Expected Int64 tensor");
    }
}

#[test]
fn test_float32_to_boolean() {
    let conf = TensorToConf {
        target_dtype: DataType::Boolean,
    };
    let mut layer = conf.to_layer().unwrap();

    // let input = TensorValue::Float32(ArrayD::from(vec![0.0, 3.14, -0.0, 1e-5]));
    let input = TensorValue::Float32(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0f32, 3.14f32, -0.0f32, 1e-5]).unwrap(),
    );
    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Boolean(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[4]);
        assert_eq!(output_arr.as_slice().unwrap(), &[false, true, false, true]);
    } else {
        panic!("Expected Boolean tensor");
    }
}

#[test]
fn test_float32_to_float16() {
    let conf = TensorToConf {
        target_dtype: DataType::Float16,
    };
    let mut layer = conf.to_layer().unwrap();

    let input =
        TensorValue::Float32(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.123456, -0.789]).unwrap());
    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Float16(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[2]);
        // 浮点精度允许误差
        let converted_back: Vec<f32> = output_arr.iter().map(|&x| x.to_f32()).collect();
        assert!((converted_back[0] - 0.123456).abs() < 0.01);
        assert!((converted_back[1] + 0.789).abs() < 0.01);
    } else {
        panic!("Expected Float16 tensor");
    }
}

#[test]
fn test_int64_to_float32() {
    let conf = TensorToConf {
        target_dtype: DataType::Float32,
    };
    let mut layer = conf.to_layer().unwrap();

    let input = TensorValue::Int64(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1i64, -2, 4]).unwrap());

    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Float32(output_arr) = &output[0] {
        assert_eq!(output_arr.as_slice().unwrap(), &[1.0, -2.0, 4.0]);
    } else {
        panic!("Expected Float32 tensor");
    }
}

#[test]
fn test_int64_to_boolean() {
    let conf = TensorToConf {
        target_dtype: DataType::Boolean,
    };
    let mut layer = conf.to_layer().unwrap();

    let input = TensorValue::Int64(ArrayD::from_shape_vec(IxDyn(&[4]), vec![0, -5, 3, 0]).unwrap());

    let output = layer.forward(&vec![input]).unwrap();

    if let TensorValue::Boolean(output_arr) = &output[0] {
        assert_eq!(output_arr.as_slice().unwrap(), &[false, true, true, false]);
    } else {
        panic!("Expected Boolean tensor");
    }
}
