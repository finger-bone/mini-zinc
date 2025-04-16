use super::{
    conf::{self, ToLayer},
    layer::Forward,
};
use crate::op::dtype::{DataType, TensorValue};
use anyhow::{Result, anyhow};
use ndarray::ArrayD;

pub struct TensorToLayer {
    pub lconf: conf::TensorToConf,
}

impl Forward for TensorToLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let input_tensor = &input[0];
        let target_dtype = &self.lconf.target_dtype;

        let converted = match input_tensor {
            TensorValue::Float32(arr) => match target_dtype {
                DataType::Int64 => {
                    let data: Vec<i64> = arr.iter().map(|&x| x as i64).collect();
                    TensorValue::Int64(ArrayD::from_shape_vec(arr.raw_dim(), data)?)
                }
                DataType::Boolean => {
                    let data: Vec<bool> = arr.iter().map(|&x| x != 0.0).collect();
                    TensorValue::Boolean(ArrayD::from_shape_vec(arr.raw_dim(), data)?)
                }
                DataType::Float16 => {
                    use half::f16;
                    let data: Vec<f16> = arr.iter().map(|&x| f16::from_f32(x)).collect();
                    TensorValue::Float16(ArrayD::from_shape_vec(arr.raw_dim(), data)?)
                }
                DataType::BFloat16 => {
                    use half::bf16;
                    let data: Vec<bf16> = arr.iter().map(|&x| bf16::from_f32(x)).collect();
                    TensorValue::BFloat16(ArrayD::from_shape_vec(arr.raw_dim(), data)?)
                }
                DataType::Float32 => input_tensor.clone(),
                // _ => return Err(anyhow!("Unsupported target dtype {:?} for Float32", target_dtype)),
            },
            TensorValue::Int64(arr) => match target_dtype {
                DataType::Float32 => {
                    let data: Vec<f32> = arr.iter().map(|&x| x as f32).collect();
                    TensorValue::Float32(ArrayD::from_shape_vec(arr.raw_dim(), data)?)
                }
                DataType::Boolean => {
                    let data: Vec<bool> = arr.iter().map(|&x| x != 0).collect();
                    TensorValue::Boolean(ArrayD::from_shape_vec(arr.raw_dim(), data)?)
                }
                DataType::Int64 => input_tensor.clone(),
                _ => {
                    return Err(anyhow!(
                        "Unsupported target dtype {:?} for Int64",
                        target_dtype
                    ));
                }
            },
            _ => return Err(anyhow!("Unsupported input type.")),
        };

        Ok(vec![converted])
    }
}

impl ToLayer for conf::TensorToConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(TensorToLayer { lconf: self }))
    }
}
