use crate::op::conf::{SoftmaxConf, ToLayer};
use crate::op::dtype::TensorValue;
use ndarray::array;

#[test]
fn test_softmax_opencl() {
    // 构造2x3输入
    let input = array![[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]].into_dyn();
    let input_tensor = TensorValue::Float32(input);
    let mut layer = SoftmaxConf {
        axis: -1,
    }.to_layer().unwrap();
    let output = layer.forward(&vec![input_tensor]).unwrap();
    if let TensorValue::Float32(out) = &output[0] {
        // 期望输出
        let expected = array![
            [0.09003057, 0.24472848, 0.66524094],
            [0.04201007, 0.11419520, 0.84379476]
        ].into_dyn();
        assert_eq!(out.shape(), &[2,3]);
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    } else {
        panic!("输出类型错误");
    }
}