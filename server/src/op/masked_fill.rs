use super::{
    conf::{self, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ndarray::parallel::prelude::*;
use ocl::ProQue;

pub struct MaskedFillLayer {
    pub lconf: conf::MaskedFillConf,
    pub pro_que: ProQue,
}

impl Forward for MaskedFillLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let TensorValue::Float32(data) = &input[0] else {
            return Err(anyhow::anyhow!("First input must be Float32"));
        };
        let TensorValue::Boolean(mask) = &input[1] else {
            return Err(anyhow::anyhow!("Second input must be Bool tensor"));
        };

        let size = data.len();
        assert_eq!(
            mask.len(),
            size,
            "Mask and data must have same element count"
        );

        let output_buffer = self.pro_que.buffer_builder::<f32>().len(size).build()?;
        let data_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(size)
            .copy_host_slice(data.as_slice().unwrap())
            .build()?;
        let mask_u8 = mask.par_iter().map(|&x| x as u8).collect::<Vec<u8>>();
        let mask_buffer = self
            .pro_que
            .buffer_builder::<u8>()
            .len(size)
            .copy_host_slice(&mask_u8)
            .build()?;

        let kernel = self
            .pro_que
            .kernel_builder("masked_fill")
            .global_work_size(size)
            .arg(&data_buffer)
            .arg(&mask_buffer)
            .arg(&output_buffer)
            .arg(self.lconf.value)
            .arg(size as u32)
            .build()?;

        unsafe { kernel.enq()? };

        let mut output = ArrayD::zeros(data.raw_dim());
        output_buffer.read(output.as_slice_mut().unwrap()).enq()?;
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for conf::MaskedFillConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(MaskedFillLayer {
            lconf: self,
            pro_que: ProQue::builder()
                .dims(512)
                .src(include_str!("./masked_fill.cl"))
                .build()
                .unwrap(),
        }))
    }
}
