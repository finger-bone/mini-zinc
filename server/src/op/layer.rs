use crate::op::dtype::TensorValue;
use anyhow::Result;

pub trait Forward: Send + Sync {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>>;
}
