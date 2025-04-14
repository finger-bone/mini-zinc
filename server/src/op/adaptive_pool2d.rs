use super::{
    conf::{self, PoolType, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct AdaptivePool2dLayer {
    pub lconf: conf::AdaptivePool2dConf,
    pub pro_que: ProQue,
}

impl Forward for AdaptivePool2dLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for AdaptivePool"));
        };
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = if input_shape.len() > 3 {
            input_shape[3]
        } else {
            1
        };

        let output_height = self.lconf.output_size[0];
        let output_width = self.lconf.output_size[1];

        // Create output buffer
        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(batch_size * channels * output_height * output_width)
            .build()
            .unwrap();

        // Create input buffer
        let input_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(input.len())
            .copy_host_slice(input.as_slice().unwrap())
            .build()
            .unwrap();

        // Build and execute kernel
        let kernel = self
            .pro_que
            .kernel_builder("adaptive_pool")
            .global_work_size(batch_size * channels * output_height * output_width) // 添加全局工作尺寸
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(batch_size as i32)
            .arg(channels as i32)
            .arg(input_height as i32)
            .arg(input_width as i32)
            .arg(output_height as i32)
            .arg(output_width as i32)
            .arg(match self.lconf.pool_type {
                PoolType::Max => 0,
                PoolType::Avg => 1,
            } as i32) // Pool type
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        // Create output array and read from buffer
        let mut output_shape = vec![batch_size, channels, output_height];
        if input_shape.len() > 3 {
            output_shape.push(output_width);
        }

        let mut output = ArrayD::zeros(ndarray::IxDyn(&output_shape));
        output_buffer
            .read(output.as_slice_mut().unwrap())
            .enq()
            .unwrap();
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for conf::AdaptivePool2dConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;

        Ok(Box::new(AdaptivePool2dLayer {
            lconf,
            pro_que: ProQue::builder()
                .src(include_str!("./adaptive_pool2d.cl"))
                .dims(512)
                .build()
                .unwrap(),
        }))
    }
}
