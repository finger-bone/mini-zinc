use crate::op::conf::{LinearConf, ToLayer};
use crate::op::dtype::TensorValue;
use ndarray::ArrayD;

#[test]
fn test_linear_forward() {
    let input = ArrayD::from_shape_vec(
        vec![2, 3], // batch=2, in_features=3
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let linear_conf = LinearConf {
        in_features: 3,
        out_features: 2,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(
                vec![2, 3], // out x in
                vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            )
            .unwrap(),
        ),
        bias: Some(TensorValue::Float32(ArrayD::from_shape_vec(vec![2], vec![0.5, -0.5]).unwrap())),
    };

    let mut layer = linear_conf.to_layer().unwrap();
    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[2, 2]);
        let expected = ArrayD::from_shape_vec(
            vec![2, 2],
            vec![
                (1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3) + 0.5,
                (1.0 * 0.4 + 2.0 * 0.5 + 3.0 * 0.6) - 0.5,
                (4.0 * 0.1 + 5.0 * 0.2 + 6.0 * 0.3) + 0.5,
                (4.0 * 0.4 + 5.0 * 0.5 + 6.0 * 0.6) - 0.5,
            ],
        )
        .unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
#[should_panic(expected = "Input features dimension must match")]
fn test_linear_incompatible_input() {
    let linear_conf = LinearConf {
        in_features: 4,
        out_features: 2,
        weights: TensorValue::Float32(ArrayD::zeros(vec![2, 4])),
        bias: Some(TensorValue::Float32(ArrayD::zeros(vec![2]))),
    };
    let input = TensorValue::Float32(ArrayD::zeros(vec![1, 3]));
    linear_conf
        .to_layer()
        .unwrap()
        .forward(&vec![input])
        .unwrap();
}

#[test]
fn test_linear_3d_input() {
    // Create a 3D input (batch=2, seq_len=3, features=4)
    let input = ArrayD::from_shape_vec(
        vec![2, 3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
    )
    .unwrap();

    let linear = LinearConf {
        in_features: 4,
        out_features: 2,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
                .unwrap(),
        ),
        bias: Some(TensorValue::Float32(ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap())),
    };

    let mut layer = linear.to_layer().unwrap();

    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();

    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[2, 3, 2]);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
fn test_linear_4d_input() {
    // Create a 4D input (batch=2, channels=2, height=2, features=3)
    let input = ArrayD::from_shape_vec(
        vec![2, 2, 2, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
    )
    .unwrap();

    let linear = LinearConf {
        in_features: 3,
        out_features: 2,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
        ),
        bias: Some(TensorValue::Float32(ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap())),
    };

    let mut layer = linear.to_layer().unwrap();

    let output = layer.forward(&vec![TensorValue::Float32(input)]).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[2, 2, 2, 2]);
    } else {
        panic!("Expected Float32 output");
    }
}

#[test]
#[should_panic(expected = "Input features dimension must match layer configuration")]
fn test_linear_invalid_features() {
    let input = ArrayD::from_shape_vec(vec![2, 3, 4], vec![1.0; 24]).unwrap();

    let linear = LinearConf {
        in_features: 3, // Mismatched with input's last dimension (4)
        out_features: 2,
        weights: TensorValue::Float32(
            ArrayD::from_shape_vec(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
        ),
        bias: Some(TensorValue::Float32(ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap())),
    };

    let mut layer = linear.to_layer().unwrap();

    layer.forward(&vec![TensorValue::Float32(input)]).unwrap(); // Should panic
}
