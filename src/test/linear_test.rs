#[cfg(test)]
mod tests {
    use ndarray::ArrayD;

    use crate::op::{conf::{FromZOpConf, LinearConf, ZOpConf}, layer::Forward};

    #[test]
    fn test_linear_forward() {
        // Create a simple 2x3 input (batch_size=2, features=3)
        let input = ArrayD::from_shape_vec(
            vec![2, 3], 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap();
        
        // Create a linear layer with 3 input features and 2 output features
        let linear = LinearConf {
            in_features: 3,
            out_features: 2,
            weights: ArrayD::from_shape_vec(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
            bias: ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap(),
        };
        
        let layer = ZOpConf::Linear(linear);
        let layer = LinearConf::from_zopconf(layer).unwrap();
        
        // Forward pass
        let output = layer.forward(&vec![input]);
        
        // Check output shape (should be 2x2)
        assert_eq!(output[0].shape(), &[2, 2]);
    }

    #[test]
    fn test_linear_3d_input() {
        // Create a 3D input (batch=2, seq_len=3, features=4)
        let input = ArrayD::from_shape_vec(
            vec![2, 3, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
            ]
        ).unwrap();

        let linear = LinearConf {
            in_features: 4,
            out_features: 2,
            weights: ArrayD::from_shape_vec(vec![2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap(),
            bias: ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap(),
        };

        let layer = ZOpConf::Linear(linear);
        let layer = LinearConf::from_zopconf(layer).unwrap();

        let output = layer.forward(&vec![input]);

        // Check output shape (should preserve batch and seq_len dimensions)
        assert_eq!(output[0].shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_linear_4d_input() {
        // Create a 4D input (batch=2, channels=2, height=2, features=3)
        let input = ArrayD::from_shape_vec(
            vec![2, 2, 2, 3],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
            ]
        ).unwrap();

        let linear = LinearConf {
            in_features: 3,
            out_features: 2,
            weights: ArrayD::from_shape_vec(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
            bias: ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap(),
        };

        let layer = ZOpConf::Linear(linear);
        let layer = LinearConf::from_zopconf(layer).unwrap();

        let output = layer.forward(&vec![input]);

        // Check output shape (should preserve all dimensions except the last one)
        assert_eq!(output[0].shape(), &[2, 2, 2, 2]);
    }

    #[test]
    #[should_panic(expected = "Input features dimension must match layer configuration")]
    fn test_linear_invalid_features() {
        let input = ArrayD::from_shape_vec(
            vec![2, 3, 4],
            vec![1.0; 24]
        ).unwrap();

        let linear = LinearConf {
            in_features: 3, // Mismatched with input's last dimension (4)
            out_features: 2,
            weights: ArrayD::from_shape_vec(vec![2, 3], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap(),
            bias: ArrayD::from_shape_vec(vec![2], vec![0.1, 0.2]).unwrap(),
        };

        let layer = ZOpConf::Linear(linear);
        let layer = LinearConf::from_zopconf(layer).unwrap();

        layer.forward(&vec![input]); // Should panic
    }
}