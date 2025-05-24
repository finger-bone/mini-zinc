use anyhow::Result;
use ndarray::ArrayD;

use crate::op::conf::{TensorSplitConf, ToLayer};
use crate::op::dtype::TensorValue;

#[test]
fn test_tensor_split_forward_simple() -> Result<()> {
    // 1. Build operator configuration
    let conf = TensorSplitConf {
        dim: 1, // Split along the second dimension (columns)
        indices: vec![2], // Split at index 2
    };

    // 2. Create operator layer
    let mut layer = conf.to_layer()?;

    // 3. Prepare input tensor(s)
    // Input: Float32 tensor of shape [2, 4]
    // [[1, 2, 3, 4],
    //  [5, 6, 7, 8]]
    let input_data = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 4]), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )?;
    let inputs = vec![TensorValue::Float32(input_data)];

    // 4. Call forward method
    let outputs = layer.forward(&inputs)?;

    // 5. Assert output tensor(s) shape and values
    // Expected to split into two tensors
    // First tensor: columns 0-1 (indices before 2)
    // Second tensor: columns 2-3 (indices from 2 to end)
    assert_eq!(outputs.len(), 2);

    // Check first output tensor
    if let TensorValue::Float32(output1) = &outputs[0] {
        assert_eq!(output1.shape(), &[2, 2]);
        assert_eq!(output1.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0, 5.0, 6.0]);
    } else {
        panic!("Expected Float32 output tensor for outputs[0]");
    }

    // Check second output tensor
    if let TensorValue::Float32(output2) = &outputs[1] {
        assert_eq!(output2.shape(), &[2, 2]);
        assert_eq!(output2.iter().copied().collect::<Vec<_>>(), vec![3.0, 4.0, 7.0, 8.0]);
    } else {
        panic!("Expected Float32 output tensor for outputs[1]");
    }

    Ok(())
}

#[test]
fn test_tensor_split_forward_multiple_indices() -> Result<()> {
    let conf = TensorSplitConf {
        dim: 0, // Split along the first dimension (rows)
        indices: vec![1, 3], // Split at row index 1, then at row index 3
    };
    let mut layer = conf.to_layer()?;
    // Input: Float32 tensor of shape [4, 2]
    // [[1, 2],
    //  [3, 4],
    //  [5, 6],
    //  [7, 8]]
    let input_data = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[4, 2]), 
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    )?;
    let inputs = vec![TensorValue::Float32(input_data)];
    let outputs = layer.forward(&inputs)?;

    // Expected 3 splits: [0,1), [1,3), [3,4)
    assert_eq!(outputs.len(), 3);

    if let TensorValue::Float32(output1) = &outputs[0] {
        assert_eq!(output1.shape(), &[1, 2]);
        assert_eq!(output1.iter().copied().collect::<Vec<_>>(), vec![1.0, 2.0]);
    }
    if let TensorValue::Float32(output2) = &outputs[1] {
        assert_eq!(output2.shape(), &[2, 2]);
        assert_eq!(output2.iter().copied().collect::<Vec<_>>(), vec![3.0, 4.0, 5.0, 6.0]);
    }
    if let TensorValue::Float32(output3) = &outputs[2] {
        assert_eq!(output3.shape(), &[1, 2]);
        assert_eq!(output3.iter().copied().collect::<Vec<_>>(), vec![7.0, 8.0]);
    }
    Ok(())
}

#[test]
fn test_tensor_split_negative_dim() -> Result<()> {
    let conf = TensorSplitConf {
        dim: -1, // Equivalent to dim 1 for a 2D tensor
        indices: vec![1],
    };
    let mut layer = conf.to_layer()?;
    let input_data = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.])?;
    let inputs = vec![TensorValue::Float32(input_data)];
    let outputs = layer.forward(&inputs)?;

    assert_eq!(outputs.len(), 2);
    if let TensorValue::Float32(output1) = &outputs[0] {
        assert_eq!(output1.shape(), &[2, 1]);
        assert_eq!(output1.iter().copied().collect::<Vec<_>>(), vec![1.0, 4.0]);
    }
    if let TensorValue::Float32(output2) = &outputs[1] {
        assert_eq!(output2.shape(), &[2, 2]);
        assert_eq!(output2.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0, 5.0, 6.0]);
    }
    Ok(())
}