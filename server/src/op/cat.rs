use super::{conf::{self, ToLayer}, layer::Forward};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};
use ndarray::Axis;

pub struct CatLayer {
    pub lconf: conf::CatConf,
}

impl Forward for CatLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        if input.is_empty() {
            return Err(anyhow::anyhow!("Cat 算子至少需要一个输入张量"));
        }
        let dim = self.lconf.dim;
        let first_dtype = input[0].dtype();
        for tensor in input.iter() {
            if tensor.dtype() != first_dtype {
                return Err(anyhow::anyhow!("Cat 算子的所有输入张量类型必须一致"));
            }
        }
        let output = match first_dtype {
            crate::op::dtype::DataType::Float32 => {
                let arrays: Vec<_> = input.iter().map(|t| t.as_float32().unwrap()).collect();
                let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
                let concatenated = ndarray::concatenate(Axis(dim as usize), &views)?;
                TensorValue::Float32(concatenated.to_owned())
            }
            crate::op::dtype::DataType::Int64 => {
                let arrays: Vec<_> = input.iter().map(|t| t.as_int64().unwrap()).collect();
                let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
                let concatenated = ndarray::concatenate(Axis(dim as usize), &views)?;
                TensorValue::Int64(concatenated.to_owned())
            }
            crate::op::dtype::DataType::Boolean => {
                let arrays: Vec<_> = input.iter().map(|t| t.as_boolean().unwrap()).collect();
                let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
                let concatenated = ndarray::concatenate(Axis(dim as usize), &views)?;
                TensorValue::Boolean(concatenated.to_owned())
            }
            crate::op::dtype::DataType::BFloat16 => {
                let arrays: Vec<_> = input.iter().map(|t| t.as_bfloat16().unwrap()).collect();
                let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
                let concatenated = ndarray::concatenate(Axis(dim as usize), &views)?;
                TensorValue::BFloat16(concatenated.to_owned())
            }
            crate::op::dtype::DataType::Float16 => {
                let arrays: Vec<_> = input.iter().map(|t| t.as_float16().unwrap()).collect();
                let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
                let concatenated = ndarray::concatenate(Axis(dim as usize), &views)?;
                TensorValue::Float16(concatenated.to_owned())
            }
        };
        Ok(vec![output])
    }
}

impl ToLayer for conf::CatConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(CatLayer { lconf: self }))
    }
}