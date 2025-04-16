use super::{
    conf::{LayerNormConf, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Result, anyhow};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct LayerNormLayer {
    pub lconf: LayerNormConf,
    pub pro_que: ProQue,
}

impl Forward for LayerNormLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow!("Unsupported input type for LayerNorm"));
        };
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let feature_size: usize = input_shape[1..].iter().product(); // Flatten features

        // Create output buffer
        let output_size = input.len();
        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(output_size)
            .build()?;

        // Create input buffer
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

        // Create gamma buffer
        let gamma_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(gamma.len())
            .copy_host_slice(gamma.as_slice().unwrap())
            .build()?;

        let beta = match &self.lconf.bias {
            TensorValue::Float32(beta) => beta,
            _ => return Err(anyhow!("Unsupported beta type for LayerNorm")),
        };

        // Create beta buffer
        let beta_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(beta.len())
            .copy_host_slice(beta.as_slice().unwrap())
            .build()?;

        // Build and execute kernel
        let kernel = self
            .pro_que
            .kernel_builder("layernorm")
            .global_work_size(output_size) // Process each element
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(&gamma_buffer)
            .arg(&beta_buffer)
            .arg(batch_size as i32)
            .arg(feature_size as i32)
            .arg(self.lconf.eps as f32)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        // Create output array and read from buffer
        let mut output = ArrayD::zeros(input.raw_dim());
        output_buffer
            .read(output.as_slice_mut().unwrap())
            .enq()?;

        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for LayerNormConf {
    // fn to_layer(self) -> Result<Box<dyn Forward>> {
    //     let lconf = self;
    //     let pro_que = ProQue::builder()
    //         .src(include_str!("./layernorm.cl"))
    //         .build()?;

    //     Ok(Box::new(LayerNormLayer {
    //         lconf,
    //         pro_que,
    //     }))
    // }
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(LayerNormLayer {
            lconf: self,
            pro_que: ProQue::builder()
                .dims(512)
                .src(include_str!("./layernorm.cl"))
                .build()
                .unwrap(),
        }))
    }
}
