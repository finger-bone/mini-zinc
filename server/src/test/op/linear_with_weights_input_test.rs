use anyhow::Result;
use ndarray::ArrayD;

use crate::op::conf::{LinearWithWeightsInputConf, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_linear_with_weights_input_forward() -> Result<()> {
    // 1. Build operator configuration
    let in_features = 3;
    let out_features = 2;
    // Assuming bias is a Float32 tensor, adjust if different
    let bias_data = ArrayD::from_shape_vec(ndarray::IxDyn(&[out_features]), vec![0.1, 0.2])?;
    let conf = LinearWithWeightsInputConf {
        in_features,
        out_features,
        bias: Some(TensorValue::Float32(bias_data)), // This matches the structure in linear.rs for bias
    };

    // 2. Create operator layer
    let mut layer = conf.to_layer()?;

    // 3. Prepare input tensor(s)
    // Input 0: Weights (Float32)
    let weights_data = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[out_features, in_features]), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )?;
    let weights_tensor = TensorValue::Float32(weights_data);

    // Input 1: Actual input (Float32)
    // Example: batch_size = 1
    let input_data_arr = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, in_features]), vec![1.0, 2.0, 3.0])?;
    let input_tensor = TensorValue::Float32(input_data_arr);

    let inputs = vec![weights_tensor, input_tensor];

    // 4. Call forward method
    let outputs = layer.forward(&inputs)?;

    // 5. Assert output tensor(s) shape and values
    assert_eq!(outputs.len(), 1);
    if let TensorValue::Float32(output_tensor) = &outputs[0] {
        // Expected shape: [batch_size, out_features]
        assert_eq!(output_tensor.shape(), &[1, out_features]);

        // Calculate expected output (manual matrix multiplication + bias)
        // W = [[1, 2, 3], [4, 5, 6]]
        // X = [[1, 2, 3]]
        // Bias = [0.1, 0.2]
        // W.X^T + B
        // Output_0 = (1*1 + 2*2 + 3*3) + 0.1 = (1 + 4 + 9) + 0.1 = 14 + 0.1 = 14.1
        // Output_1 = (4*1 + 5*2 + 6*3) + 0.2 = (4 + 10 + 18) + 0.2 = 32 + 0.2 = 32.2
        let expected_output_data = vec![14.1, 32.2];
        assert_eq!(output_tensor.iter().map(|&x| (x * 10.0).round() / 10.0).collect::<Vec<_>>(), expected_output_data);
    } else {
        panic!("Expected Float32 output tensor");
    }

    Ok(())
}