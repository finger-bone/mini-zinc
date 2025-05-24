use super::{
    conf::{self, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};
use ndarray::IxDyn;

pub struct UnsqueezeLayer {
    pub lconf: conf::UnsqueezeConf,
}

impl Forward for UnsqueezeLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let input_tensor = &input[0];
        let mut axes = self.lconf.axes.clone();
        // convert negative axis to positive axis
        for i in 0..axes.len() {
            if axes[i] < 0 {
                axes[i] = axes[i] + input_tensor.shape().len() as isize + 1;
            }
        }
        axes.sort();
        let mut shape = match input_tensor {
            TensorValue::Float32(arr) => arr.shape().to_vec(),
            TensorValue::Int64(arr) => arr.shape().to_vec(),
            TensorValue::Boolean(arr) => arr.shape().to_vec(),
            TensorValue::BFloat16(arr) => arr.shape().to_vec(),
            TensorValue::Float16(arr) => arr.shape().to_vec(),
        };
        for &axis in axes.iter() {
            shape.insert(axis as usize, 1);
        }
        let new_shape = IxDyn(&shape);
        let output = match input_tensor {
            TensorValue::Float32(arr) => {
                TensorValue::Float32(arr.clone().to_shape(new_shape)?.to_owned())
            }
            TensorValue::Int64(arr) => {
                TensorValue::Int64(arr.clone().to_shape(new_shape)?.to_owned())
            }
            TensorValue::Boolean(arr) => {
                TensorValue::Boolean(arr.clone().to_shape(new_shape)?.to_owned())
            }
            TensorValue::BFloat16(arr) => {
                TensorValue::BFloat16(arr.clone().to_shape(new_shape)?.to_owned())
            }
            TensorValue::Float16(arr) => {
                TensorValue::Float16(arr.clone().to_shape(new_shape)?.to_owned())
            }
        };
        Ok(vec![output])
    }
}

impl ToLayer for conf::UnsqueezeConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(UnsqueezeLayer { lconf: self }))
    }
}
