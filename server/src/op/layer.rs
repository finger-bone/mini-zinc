use crate::op::dtype::TensorValue;
use anyhow::Result;

pub trait Forward {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>>;
}
