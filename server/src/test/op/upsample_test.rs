use crate::op::conf::UpSampleMode;

#[test]
fn test_upsample_nearest() {
    use crate::op::conf::{UpSampleConf, ToLayer};
    use crate::op::dtype::TensorValue;
    use ndarray::array;
    // 构造 1x1x2x2 输入
    let input = array![[[[1.0, 2.0], [3.0, 4.0]]]].into_dyn();
    let input_tensor = TensorValue::Float32(input);
    let conf = UpSampleConf {
        mode: UpSampleMode::Nearest,
        scale_factor: Some(vec![2.0, 2.0]),
        size: None,
    };
    let mut layer = conf.to_layer().unwrap();
    let output = layer.forward(&vec![input_tensor]).unwrap();
    if let TensorValue::Float32(out) = &output[0] {
        // 期望输出为 1x1x4x4，且每个2x2块与原输入对应
        let expected = array![
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0]
            ]
        ].into_dyn();
        assert_eq!(out.shape(), &[1,1,4,4]);
        let diff = out - &expected;
        let diff = diff.mapv(|x| x.abs());
        assert!(diff.sum() < 1e-3);
    } else {
        panic!("输出类型错误");
    }
}