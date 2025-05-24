use anyhow::Result;
use ndarray::ArrayD;

use crate::op::conf::{ScalerEqConf, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_scaler_eq_forward() -> Result<()> {
    // 1. Build operator configuration
    let conf = ScalerEqConf {
        other: 5.0, // The scalar value to compare against
    };

    // 2. Create operator layer
    let mut layer = conf.to_layer()?;

    // 3. Prepare input tensor(s)
    // Input: Float32 tensor
    let input_data =
        ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 5.0, 5.000001, 4.999999])?;
    let inputs = vec![TensorValue::Float32(input_data)];

    // 4. Call forward method
    let outputs = layer.forward(&inputs)?;

    // 5. Assert output tensor(s) shape and values
    assert_eq!(outputs.len(), 1);
    if let TensorValue::Boolean(output_tensor) = &outputs[0] {
        // Expected shape: same as input
        assert_eq!(output_tensor.shape(), &[2, 2]);

        // Expected output: boolean tensor where true if element is close to `other`
        // Based on the default eps = 1e-5f32 in scale_eq.rs
        // 1.0 is not 5.0 -> false
        // 5.0 is 5.0 -> true
        // 5.000001 is 5.0 ( (5.000001 - 5.0).abs() < 1e-5 ) -> true
        // 4.999999 is 5.0 ( (4.999999 - 5.0).abs() < 1e-5 ) -> true
        let expected_output_data = vec![false, true, true, true];
        assert_eq!(
            output_tensor.iter().copied().collect::<Vec<_>>(),
            expected_output_data
        );
    } else {
        panic!("Expected Boolean output tensor");
    }

    Ok(())
}

#[test]
fn test_scale_eq_forward_no_match() -> Result<()> {
    let conf = ScalerEqConf { other: 10.0 };
    let mut layer = conf.to_layer()?;
    let input_data = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0])?;
    let inputs = vec![TensorValue::Float32(input_data)];
    let outputs = layer.forward(&inputs)?;

    assert_eq!(outputs.len(), 1);
    if let TensorValue::Boolean(output_tensor) = &outputs[0] {
        assert_eq!(output_tensor.shape(), &[1, 3]);
        let expected_output_data = vec![false, false, false];
        assert_eq!(
            output_tensor.iter().copied().collect::<Vec<_>>(),
            expected_output_data
        );
    } else {
        panic!("Expected Boolean output tensor");
    }
    Ok(())
}
