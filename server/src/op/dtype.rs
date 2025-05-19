use half::{bf16, f16};
use ndarray::ArrayD;

#[derive(Debug, PartialEq)]
pub enum DataType {
    BFloat16,
    Float16,
    Float32,
    Boolean,
    Int64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TensorValue {
    BFloat16(ArrayD<bf16>),
    Float16(ArrayD<f16>),
    Float32(ArrayD<f32>),
    Boolean(ArrayD<bool>),
    Int64(ArrayD<i64>),
}

impl TensorValue {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            TensorValue::BFloat16(arr) => arr.shape().to_vec(),
            TensorValue::Float16(arr) => arr.shape().to_vec(),
            TensorValue::Float32(arr) => arr.shape().to_vec(),
            TensorValue::Boolean(arr) => arr.shape().to_vec(),
            TensorValue::Int64(arr) => arr.shape().to_vec(),
        }
    }
}