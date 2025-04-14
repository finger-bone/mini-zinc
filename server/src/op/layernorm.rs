use super::{
    conf::{LayerNormConf, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Result, anyhow};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct LayerNormLayer {
    pub lconf: LayerNormConf,
    pub pro_que: ProQue,
    pub gamma: ocl::Buffer<f32>,
    pub beta: ocl::Buffer<f32>,
}

impl Forward for LayerNormLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let TensorValue::Float32(input_arr) = &input[0] else {
            return Err(anyhow!("Expected Float32 input for LayerNorm"));
        };

        let input_shape = input_arr.shape();
        assert_eq!(
            input_shape.len(),
            3,
            "LayerNorm only supports 3D input: [batch, seq_len, embed_dim]"
        );

        let (batch_size, seq_len, embed_dim) = (input_shape[0], input_shape[1], input_shape[2]);
        let total_samples = batch_size * seq_len;

        let input_flat = input_arr
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get input slice"))?;

        let input_buffer = self
            .pro_que
            .buffer_builder()
            .len(input_flat.len())
            .copy_host_slice(input_flat)
            .build()?;

        let output_buffer = self
            .pro_que
            .buffer_builder()
            .len(input_flat.len())
            .build()?;

        let kernel = self
            .pro_que
            .kernel_builder("layernorm")
            .global_work_size(input_flat.len())
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(&self.gamma)
            .arg(&self.beta)
            .arg(self.lconf.eps)
            .arg(embed_dim as i32)
            .arg(total_samples as i32)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        let mut output_data = vec![0f32; input_flat.len()];
        output_buffer.read(&mut output_data).enq()?;

        let output_arr = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[batch_size, seq_len, embed_dim]),
            output_data,
        )?;

        Ok(vec![TensorValue::Float32(output_arr)])
    }
}

impl ToLayer for LayerNormConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let embed_dim = self.normalized_shape[0];

        let gamma_arr = match &self.weight {
            TensorValue::Float32(arr) => arr.clone(),
            _ => return Err(anyhow!("Gamma must be Float32")),
        };
        let beta_arr = match &self.bias {
            TensorValue::Float32(arr) => arr.clone(),
            _ => return Err(anyhow!("Beta must be Float32")),
        };

        let pro_que = ProQue::builder()
            .dims(embed_dim)
            .src(include_str!("layernorm.cl"))
            .build()?;

        let gamma_buf = pro_que
            .buffer_builder()
            .len(embed_dim)
            .copy_host_slice(gamma_arr.as_slice().unwrap())
            .build()?;

        let beta_buf = pro_que
            .buffer_builder()
            .len(embed_dim)
            .copy_host_slice(beta_arr.as_slice().unwrap())
            .build()?;

        Ok(Box::new(LayerNormLayer {
            lconf: self,
            pro_que,
            gamma: gamma_buf,
            beta: beta_buf,
        }))
    }
}
