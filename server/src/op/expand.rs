use anyhow::Result;
use ndarray::prelude::*;

use crate::op::dtype::TensorValue;

use super::{conf::{self, ExpandConf, ToLayer}, layer::Forward};

pub struct ExpandLayer {
    pub lconf: conf::ExpandConf,
}

impl Forward for ExpandLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let target_shape = IxDyn(&self.lconf.shape);
        
        match &input[0] {
            TensorValue::Float32(input_arr) => {
                let broadcasted = input_arr.broadcast(target_shape.clone()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Cannot broadcast input shape {:?} to target shape {:?}",
                        input_arr.shape(),
                        target_shape
                    )
                })?;
                // Clone to make it into a real owned ArrayD (since broadcast returns a view)
                let output = broadcasted.to_owned();
                Ok(vec![TensorValue::Float32(output)])
            },
            TensorValue::BFloat16(input_arr) => {
                let broadcasted = input_arr.broadcast(target_shape.clone()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Cannot broadcast input shape {:?} to target shape {:?}",
                        input_arr.shape(),
                        target_shape
                    )
                })?;
                let output = broadcasted.to_owned();
                Ok(vec![TensorValue::BFloat16(output)])
            },
            TensorValue::Float16(input_arr) => {
                let broadcasted = input_arr.broadcast(target_shape.clone()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Cannot broadcast input shape {:?} to target shape {:?}",
                        input_arr.shape(),
                        target_shape
                    )
                })?;
                let output = broadcasted.to_owned();
                Ok(vec![TensorValue::Float16(output)])
            },
            TensorValue::Boolean(input_arr) => {
                let broadcasted = input_arr.broadcast(target_shape.clone()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Cannot broadcast input shape {:?} to target shape {:?}",
                        input_arr.shape(),
                        target_shape
                    )
                })?;
                let output = broadcasted.to_owned();
                Ok(vec![TensorValue::Boolean(output)])
            },
            TensorValue::Int64(input_arr) => {
                let broadcasted = input_arr.broadcast(target_shape.clone()).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Cannot broadcast input shape {:?} to target shape {:?}",
                        input_arr.shape(),
                        target_shape
                    )
                })?;
                let output = broadcasted.to_owned();
                Ok(vec![TensorValue::Int64(output)])
            }
        }
    }
}

impl ToLayer for ExpandConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(ExpandLayer { lconf: self }))
    }
}
