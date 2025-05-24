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

    pub fn dtype(&self) -> DataType {
        match self {
            TensorValue::BFloat16(_) => DataType::BFloat16,
            TensorValue::Float16(_) => DataType::Float16,
            TensorValue::Float32(_) => DataType::Float32,
            TensorValue::Boolean(_) => DataType::Boolean,
            TensorValue::Int64(_) => DataType::Int64,
        }
    }

    pub fn as_float32(&self) -> Option<&ArrayD<f32>> {
        if let TensorValue::Float32(arr) = self {
            Some(arr)
        } else {
            None
        }
    }
    pub fn as_int64(&self) -> Option<&ArrayD<i64>> {
        if let TensorValue::Int64(arr) = self {
            Some(arr)
        } else {
            None
        }
    }
    pub fn as_boolean(&self) -> Option<&ArrayD<bool>> {
        if let TensorValue::Boolean(arr) = self {
            Some(arr)
        } else {
            None
        }
    }
    pub fn as_bfloat16(&self) -> Option<&ArrayD<bf16>> {
        if let TensorValue::BFloat16(arr) = self {
            Some(arr)
        } else {
            None
        }
    }
    pub fn as_float16(&self) -> Option<&ArrayD<f16>> {
        if let TensorValue::Float16(arr) = self {
            Some(arr)
        } else {
            None
        }
    }
}
