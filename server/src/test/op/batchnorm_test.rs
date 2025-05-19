use crate::op::{
    conf::{BatchNormConf, ToLayer},
    layer::{Forward, TensorValue},
};
use ndarray::ArrayD;

#[test]
fn test_batchnorm_forward() {
    let input = ArrayD::from_shape_vec(
        vec![2, 3], // batch=2, features=3
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let bn_conf = BatchNormConf { num_features: 3 };
    let layer = bn_conf.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[2, 3]);
        // 验证标准化后的均值接近0，方差接近1
        for c in 0..3 {
            let mean: f32 = output.index_axis(ndarray::Axis(1), c).iter().sum::<f32>() / 2.0;
            assert!((mean).abs() < 1e-6);
            let var = output.index_axis(ndarray::Axis(1), c)
                .iter()
                .map(|x| x*x)
                .sum::<f32>() / 2.0;
            assert!((var - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
#[should_panic(expected = "Invalid input dimensions")]
fn test_batchnorm_invalid_shape() {
    let bn_conf = BatchNormConf { num_features: 4 };
    let input = TensorValue::Float32(ArrayD::zeros(vec![2,3]));
    bn_conf.to_layer().unwrap().forward(&vec![input]);
}