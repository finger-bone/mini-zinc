use anyhow::Result;
use half::{bf16, f16};
use ndarray::ArrayD;


pub trait Forward {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TensorValue {
    Int64(ArrayD<i64>),
    Float32(ArrayD<f32>),
    Float16(ArrayD<f16>),
    BFloat16(ArrayD<bf16>),
}
