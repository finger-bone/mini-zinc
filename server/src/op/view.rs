use super::{
    conf::{self, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};

pub struct ViewLayer {
    pub lconf: conf::ViewConf,
}

impl Forward for ViewLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for View"));
        };

        // Check if the total number of elements matches
        let input_size: usize = self.lconf.input_shape.iter().product();
        let output_size: usize = self.lconf.output_shape.iter().product();
        assert_eq!(
            input_size, output_size,
            "Input and output shapes must have the same number of elements"
        );

        // Reshape the input tensor to the desired output shape
        let output = input
            .clone()
            .into_shape_with_order(ndarray::IxDyn(&self.lconf.output_shape))
            .unwrap();
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for conf::ViewConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;
        Ok(Box::new(ViewLayer { lconf }))
    }
}
