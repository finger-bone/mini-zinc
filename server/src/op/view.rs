use super::{
    conf::{self, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};

pub struct ViewLayer {
    pub lconf: conf::ViewConf,
}

impl Forward for ViewLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let input_tensor = &input[0];

        // Common function to check size and reshape
        let check_and_reshape = |shape: &[usize]| {
            let input_size: usize = shape.iter().product();
            let output_size: usize = self.lconf.output_shape.iter().product();
            assert_eq!(
                input_size, output_size,
                "Input and output shapes must have the same number of elements"
            );
            ndarray::IxDyn(&self.lconf.output_shape)
        };

        // Process based on input tensor type
        let result = match input_tensor {
            TensorValue::Float32(input) => {
                let new_shape = check_and_reshape(input.shape());
                // 确保输出张量是内存连续的
                let output = input
                    .clone()
                    .into_shape_with_order(new_shape)
                    .unwrap()
                    .to_owned();
                TensorValue::Float32(output)
            }
            TensorValue::Int64(input) => {
                let new_shape = check_and_reshape(input.shape());
                // 确保输出张量是内存连续的
                let output = input
                    .clone()
                    .into_shape_with_order(new_shape)
                    .unwrap()
                    .to_owned();
                TensorValue::Int64(output)
            }
            TensorValue::Boolean(input) => {
                let new_shape = check_and_reshape(input.shape());
                // 确保输出张量是内存连续的
                let output = input
                    .clone()
                    .into_shape_with_order(new_shape)
                    .unwrap()
                    .to_owned();
                TensorValue::Boolean(output)
            }
            TensorValue::Float16(input) => {
                let new_shape = check_and_reshape(input.shape());
                // 确保输出张量是内存连续的
                let output = input
                    .clone()
                    .into_shape_with_order(new_shape)
                    .unwrap()
                    .to_owned();
                TensorValue::Float16(output)
            }
            TensorValue::BFloat16(input) => {
                let new_shape = check_and_reshape(input.shape());
                // 确保输出张量是内存连续的
                let output = input
                    .clone()
                    .into_shape_with_order(new_shape)
                    .unwrap()
                    .to_owned();
                TensorValue::BFloat16(output)
            }
        };

        Ok(vec![result])
    }
}

impl ToLayer for conf::ViewConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;
        Ok(Box::new(ViewLayer { lconf }))
    }
}
