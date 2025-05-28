use super::{conf::{SigmoidConf, ToLayer}, layer::Forward};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};
use ocl::ProQue;

pub struct SigmoidLayer {
    pub pro_que: ProQue,
}

impl Forward for SigmoidLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for Sigmoid"));
        };
        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(input.len())
            .build()
            .unwrap();
        let input_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(input.len())
            .copy_host_slice(input.as_slice().unwrap())
            .build()
            .unwrap();
        let kernel = self
            .pro_que
            .kernel_builder("sigmoid")
            .global_work_size(input.len())
            .arg(&input_buffer)
            .arg(&output_buffer)
            .build()
            .unwrap();
        unsafe {
            kernel.enq().unwrap();
        }
        let mut output = ndarray::ArrayD::zeros(input.raw_dim());
        output_buffer
            .read(output.as_slice_mut().unwrap())
            .enq()
            .unwrap();
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for SigmoidConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(SigmoidLayer {
            pro_que: ProQue::builder()
                .dims(256)
                .src(format!(
                    "#define TILE_SIZE 32\n{}",
                    include_str!("./sigmoid.cl")
                ))
                .build()
                .unwrap(),
        }))
    }
}