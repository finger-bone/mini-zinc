use anyhow::Result;
use ocl::ProQue;
use ndarray::{ArrayD, IxDyn};
use crate::op::dtype::TensorValue;
use super::{conf::{UpSampleConf, ToLayer}, layer::Forward};

pub struct UpSampleLayer {
    pub lconf: UpSampleConf,
    pub pro_que: ProQue,
}

impl Forward for UpSampleLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let input_tensor = &input[0];
        let input_shape = input_tensor.shape();
        let input_rank = input_shape.len();
        // let mode = self.lconf.mode;
        if input_rank != 4 && input_rank != 3 {
            return Err(anyhow::anyhow!("UpSample only supports 3D/4D tensors (CHW/NCHW)"));
        }
        let (out_h, out_w) = if let Some(ref size) = self.lconf.size {
            (size[size.len()-2], size[size.len()-1])
        } else if let Some(ref scale) = self.lconf.scale_factor {
            let h = (input_shape[input_rank-2] as f32 * scale[0]).round() as usize;
            let w = (input_shape[input_rank-1] as f32 * scale[1]).round() as usize;
            (h, w)
        } else {
            return Err(anyhow::anyhow!("UpSample requires either size or scale_factor"));
        };
        let (in_h, in_w) = (input_shape[input_rank-2], input_shape[input_rank-1]);
        let output_shape: Vec<usize> = input_shape[..input_rank-2]
            .iter().cloned().chain([out_h, out_w]).collect();
        match input_tensor {
            TensorValue::Float32(arr) => {
                let arr = arr.to_owned().into_dimensionality::<ndarray::IxDyn>().unwrap();
                let mut output = ArrayD::<f32>::zeros(IxDyn(&output_shape));
                let (batch_size, channels) = if input_rank == 4 {
                    (input_shape[0], input_shape[1])
                } else {
                    (1, input_shape[0])
                };
                let in_h = in_h;
                let in_w = in_w;
                let out_h = out_h;
                let out_w = out_w;
                let input_flat = arr.as_slice().unwrap();
                let output_flat = output.as_slice_mut().unwrap();
                let pro_que = &self.pro_que;
                let input_buffer = pro_que.buffer_builder::<f32>()
                    .len(input_flat.len())
                    .copy_host_slice(input_flat)
                    .build()?;
                let output_buffer = pro_que.buffer_builder::<f32>()
                    .len(output_flat.len())
                    .build()?;
                let kernel = pro_que.kernel_builder("upsample_nearest")
                    .global_work_size(output_flat.len())
                    .arg(&input_buffer)
                    .arg(&output_buffer)
                    .arg(batch_size as i32)
                    .arg(channels as i32)
                    .arg(in_h as i32)
                    .arg(in_w as i32)
                    .arg(out_h as i32)
                    .arg(out_w as i32)
                    .build()?;
                unsafe {
                    kernel.enq()?;
                }
                output_buffer.read(output_flat).enq()?;
                Ok(vec![TensorValue::Float32(output)])
            }
            _ => Err(anyhow::anyhow!("UpSample 目前只支持 Float32 格式输入")),
        }
    }
}

impl ToLayer for UpSampleConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let pro_que = ProQue::builder()
            .src(include_str!("./upsample.cl"))
            .dims(512)
            .build()?;
        Ok(Box::new(UpSampleLayer { lconf: self, pro_que }))
    }
}