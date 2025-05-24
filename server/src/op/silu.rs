use super::{
    conf::{self, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};
use ocl::ProQue;

pub struct SiLULayer {
    pub lconf: conf::SiLUConf,
    pub pro_que: ProQue,
}

impl Forward for SiLULayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for SiLU"));
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
            .kernel_builder("silu")
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

impl ToLayer for conf::SiLUConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;
        Ok(Box::new(SiLULayer {
            lconf,
            pro_que: ProQue::builder()
                .dims(512)
                .src(format!("#define TILE_SIZE 32\n{}", include_str!("./silu.cl")))
                .build()
                .unwrap(),
        }))
    }
}