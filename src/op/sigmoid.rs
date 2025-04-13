use super::{
    conf::{self, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};
use ocl::ProQue;

pub struct SigmoidLayer {
    pub pro_que: ProQue,
}

impl Forward for SigmoidLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
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

impl ToLayer for conf::SigmoidConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let _ = self;
        Ok(Box::new(SigmoidLayer {
            pro_que: ProQue::builder()
                .src(include_str!("./sigmoid.cl"))
                .dims(256)
                .build()
                .unwrap(),
        }))
    }
}
