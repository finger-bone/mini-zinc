use super::{
    conf::{self, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};
use ocl::ProQue;

pub struct SoftmaxLayer {
    pub lconf: conf::SoftmaxConf,
    pub pro_que: ProQue,
}

impl Forward for SoftmaxLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for Softmax"));
        };
        let shape = input.shape();
        let size = input.len();

        // 确定axis维度的大小
        let axis = if self.lconf.axis < 0 {
            (shape.len() as i32 + self.lconf.axis) as usize
        } else {
            self.lconf.axis as usize
        };
        let axis_size = shape[axis];

        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(size)
            .build()
            .unwrap();
        let input_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(size)
            .copy_host_slice(input.as_slice().unwrap())
            .build()
            .unwrap();

        let kernel = self
            .pro_que
            .kernel_builder("softmax")
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(size as i32)
            .arg(axis_size as i32)
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

impl ToLayer for conf::SoftmaxConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;
        Ok(Box::new(SoftmaxLayer {
            lconf,
            pro_que: ProQue::builder()
                .src(include_str!("./softmax.cl"))
                .dims(256)
                .build()
                .unwrap(),
        }))
    }
}
