use anyhow::Result;
use ocl::ProQue;
use ndarray::{ArrayD, IxDyn};
use crate::op::dtype::TensorValue;
use super::{conf::{SoftmaxConf, ToLayer}, layer::Forward};

pub struct SoftmaxLayer {
    pub axis: isize,
    pub pro_que: ProQue,
}

impl Forward for SoftmaxLayer {
    fn forward(&mut self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let input_tensor = &input[0];
        match input_tensor {
            TensorValue::Float32(arr) => {
                let arr = arr.to_owned().into_dimensionality::<ndarray::IxDyn>().unwrap();
                let axis = if self.axis < 0 {
                    ((arr.ndim() as isize) + self.axis) as usize                    
                } else {
                    self.axis as usize
                };
                let shape = arr.shape();
                let n_axis = shape[axis];
                let batch = arr.len() / n_axis;
                let input_flat = arr.view().into_shape_with_order((batch, n_axis)).unwrap();
                let mut output = ArrayD::<f32>::zeros(IxDyn(arr.shape()));
                let mut output_flat = output.view_mut().into_shape_with_order((batch, n_axis)).unwrap();
                // CPU端计算max_val
                let mut max_vals = Vec::with_capacity(batch);
                for i in 0..batch {
                    let row = &input_flat.row(i);
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    max_vals.push(max);
                }
                let pro_que = &self.pro_que;
                let input_buf = pro_que.buffer_builder::<f32>()
                    .len(input_flat.len())
                    .copy_host_slice(input_flat.as_slice().unwrap())
                    .build()?;
                let output_buf = pro_que.buffer_builder::<f32>()
                    .len(output_flat.len())
                    .build()?;
                let maxval_buf = pro_que.buffer_builder::<f32>()
                    .len(max_vals.len())
                    .copy_host_slice(&max_vals)
                    .build()?;
                let kernel = pro_que.kernel_builder("softmax_safe")
                    .global_work_size(batch)
                    .arg(&input_buf)
                    .arg(&output_buf)
                    .arg(&maxval_buf)
                    .arg(batch as i32)
                    .arg(n_axis as i32)
                    .build()?;
                unsafe { kernel.enq()?; }
                output_buf.read(output_flat.as_slice_mut().unwrap()).enq()?;
                Ok(vec![TensorValue::Float32(output)])
            }
            _ => Err(anyhow::anyhow!("Softmax 目前只支持 Float32 格式输入")),
        }
    }
}

impl ToLayer for SoftmaxConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let pro_que = ProQue::builder()
            .src(include_str!("softmax.cl"))
            .dims(1)
            .build()
            .unwrap();
        Ok(Box::new(SoftmaxLayer {
            axis: self.axis,
            pro_que,
        }))
    }
}