use super::{
    conf::{self, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};
use ndarray::IxDyn;

pub struct FlattenLayer {
    pub fconf: conf::FlattenConf,
}

impl Forward for FlattenLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for Flatten"));
        };

        let input_shape = input.shape();
        let ndim = input_shape.len();

        // Handle negative dimensions
        let start_dim = if self.fconf.start_dim < 0 {
            (ndim as isize + self.fconf.start_dim) as usize
        } else {
            self.fconf.start_dim as usize
        };

        let end_dim = if self.fconf.end_dim < 0 {
            (ndim as isize + self.fconf.end_dim) as usize
        } else {
            self.fconf.end_dim as usize
        };

        // Validate dimensions
        assert!(
            start_dim <= end_dim && end_dim < ndim,
            "Invalid start_dim or end_dim for Flatten"
        );

        // Compute flattened shape
        let mut output_shape: Vec<usize> = vec![];
        output_shape.extend_from_slice(&input_shape[..start_dim]);
        let flatten_size: usize = input_shape[start_dim..=end_dim].iter().product();
        output_shape.push(flatten_size);
        output_shape.extend_from_slice(&input_shape[end_dim + 1..]);

        // Reshape the input tensor
        let output = input
            .clone()
            .into_shape_with_order(IxDyn(&output_shape))
            .unwrap();
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for conf::FlattenConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let fconf = self;
        Ok(Box::new(FlattenLayer { fconf }))
    }
}
