use super::{
    conf::{RSMNormConf, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Result, anyhow};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct RSMNormLayer {
    pub lconf: RSMNormConf,
    pub pro_que: ProQue,
}

impl Forward for RSMNormLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow!("Unsupported input type for RSMNorm"));
        };
        let input_shape = input.shape();
        let output_size = input.len();
        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(output_size)
            .build()?;
        let input_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(input.len())
            .copy_host_slice(input.as_slice().unwrap())
            .build()?;
        let gamma = match &self.lconf.weight {
            TensorValue::Float32(gamma) => gamma,
            _ => return Err(anyhow!("Unsupported type for weight.")),
        };
        let gamma_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(gamma.len())
            .copy_host_slice(gamma.as_slice().unwrap())
            .build()?;

        let batch = input_shape[0] as i32;
        let inner = input_shape[1..input_shape.len() - 1].iter().product::<usize>() as i32;
        let channel = input_shape[input_shape.len() - 1] as i32;
        let kernel = self
            .pro_que
            .kernel_builder("rsmnorm")
            .global_work_size(output_size)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(&gamma_buffer)
            .arg(batch)
            .arg(inner)
            .arg(channel)
            .arg(self.lconf.eps as f32)
            .build()?;
        unsafe {
            kernel.enq()?;
        }
        let mut output = ArrayD::zeros(input.raw_dim());
        output_buffer.read(output.as_slice_mut().unwrap()).enq()?;
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for RSMNormConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(RSMNormLayer {
            lconf: self,
            pro_que: ProQue::builder()
                .dims(512)
                .src(include_str!("./rsmnorm.cl"))
                .build()
                .unwrap(),
        }))
    }
}
