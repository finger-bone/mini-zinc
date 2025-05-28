use crate::op::sigmoid::SigmoidLayer;
use crate::op::layer::Forward;
use crate::op::dtype::TensorValue;
use ndarray::ArrayD;

#[test]
fn test_sigmoid_opencl() {
    let input = TensorValue::Float32(ArrayD::from_shape_vec(vec![2, 2], vec![0.0, 1.0, -1.0, 2.0]).unwrap());
    let mut layer = SigmoidLayer {
        pro_que: ocl::ProQue::builder()
            .dims(256)
            .src(format!("#define TILE_SIZE 32\n{}", include_str!("../../op/sigmoid.cl")))
            .build()
            .unwrap(),
    };
    let output = layer.forward(&vec![input]).unwrap();
    if let TensorValue::Float32(arr) = &output[0] {
        let expected = [0.5, 1.0/(1.0+(-1.0f32).exp()), 1.0/(1.0+1.0f32.exp()), 1.0/(1.0+(-2.0f32).exp())];
        for (o, e) in arr.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "sigmoid output mismatch: {} vs {}", o, e);
        }
    } else {
        panic!("Expected Float32 output");
    }
}