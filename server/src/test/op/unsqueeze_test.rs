use crate::op::conf::{ToLayer, UnsqueezeConf};
use crate::op::dtype::TensorValue;
use ndarray::ArrayD;

#[test]
fn test_unsqueeze_forward() {
    // 在第0维和第2维插入新轴
    let unsqueeze = UnsqueezeConf { axes: vec![0, 2] };
    let mut layer = unsqueeze.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![3, 4], (1..=12).map(|x| x as f32).collect()).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    if let TensorValue::Float32(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[1, 3, 1, 4]);
        assert_eq!(output_arr.len(), input.len());
        assert_eq!(
            output_arr.iter().cloned().collect::<Vec<_>>(),
            input.iter().cloned().collect::<Vec<_>>()
        );
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_unsqueeze_no_axes() {
    // 不插入任何轴，输出应与输入一致
    let unsqueeze = UnsqueezeConf { axes: vec![] };
    let mut layer = unsqueeze.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    if let TensorValue::Float32(output_arr) = &output[0] {
        assert_eq!(output_arr.shape(), &[2, 2]);
        assert_eq!(output_arr, &input);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
#[should_panic]
fn test_unsqueeze_invalid_axis() {
    // 插入超出范围的轴，应 panic
    let unsqueeze = UnsqueezeConf { axes: vec![10] };
    let mut layer = unsqueeze.to_layer().unwrap();
    let input = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    layer.forward(&vec![TensorValue::Float32(input)]).unwrap();
}
