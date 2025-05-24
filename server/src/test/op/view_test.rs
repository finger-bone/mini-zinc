use crate::op::conf::{ToLayer, ViewConf};
use crate::op::dtype::TensorValue;
use ndarray::ArrayD;

#[test]
fn test_view_forward() {
    // 测试2D到1D的重塑
    let view = ViewConf {
        // input_shape: vec![2, 3],
        output_shape: vec![6],
    };
    let mut layer = view.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    let expected = ArrayD::from_shape_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output, expected);
    } else {
        panic!("Output is not Float32");
    }

    // 测试3D到2D的重塑
    let view = ViewConf {
        // input_shape: vec![2, 2, 2],
        output_shape: vec![4, 2],
    };
    let mut layer = view.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .unwrap();
    let output = layer
        .forward(&vec![TensorValue::Float32(input.clone())])
        .unwrap();
    let expected =
        ArrayD::from_shape_vec(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output, expected);
    } else {
        panic!("Output is not Float32");
    }
}

#[test]
#[should_panic(expected = "Input and output shapes must have the same number of elements")]
fn test_view_shape_mismatch() {
    let view = ViewConf {
        // input_shape: vec![2, 3],
        output_shape: vec![5], // 元素数量不匹配
    };
    let mut layer = view.to_layer().unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    layer.forward(&vec![TensorValue::Float32(input)]).unwrap(); // 这里应该会panic
}
