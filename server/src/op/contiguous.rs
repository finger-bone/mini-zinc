use super::{conf::{self, ToLayer}, dtype::TensorValue, layer::Forward};

impl ToLayer for conf::ContiguousConf {
    fn to_layer(self: Self) -> anyhow::Result<Box<dyn super::layer::Forward>> {
        Ok(Box::new(ContiguousLayer {

        }))
    }
}

pub struct ContiguousLayer {
    
}

impl Forward for ContiguousLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> anyhow::Result<Vec<TensorValue>> {
        // take the first input
        let input = &input[0];
        // clone the input
        let output = input.clone();
        Ok(vec![output.to_owned()])
    }
}