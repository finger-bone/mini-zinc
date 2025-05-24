use anyhow::Result;
use ndarray::ArrayD;

use crate::op::conf::{ContiguousConf, ToLayer}; // Assuming ContiguousConf exists
use crate::op::dtype::TensorValue;

#[test]
fn test_contiguous_forward() -> Result<()> {
    // 1. Build operator configuration
    // TODO: Replace with actual configuration for ContiguousConf if it takes parameters
    let conf = ContiguousConf {};

    // 2. Create operator layer
    let mut layer = conf.to_layer()?;

    // 3. Prepare input tensor(s)
    // TODO: Replace with actual input tensor(s) for the contiguous operator
    // Example: Creating a simple Float32 tensor. Adjust shape and data as needed.
    let input_data = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0])?;
    let inputs = vec![TensorValue::Float32(input_data)];

    // 4. Call forward method
    let outputs = layer.forward(&inputs)?;

    // 5. Assert output tensor(s) shape and values
    // TODO: Replace with actual assertions based on the expected behavior of the contiguous operator
    assert_eq!(outputs.len(), 1);
    if let TensorValue::Float32(output_tensor) = &outputs[0] {
        // Example: Asserting the shape. Adjust as needed.
        assert_eq!(output_tensor.shape(), &[1, 2, 2]);
        // Example: Asserting some values. Adjust as needed.
        // This assumes contiguous might just return the same tensor if already contiguous
        // or a new contiguous tensor with the same data.
        assert_eq!(
            output_tensor.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    } else {
        panic!("Expected Float32 output tensor");
    }

    Ok(())
}
