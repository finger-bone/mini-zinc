use super::{
    conf::{self, PoolType, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct Pool2dLayer {
    pub lconf: conf::Pool2dConf,
    pub pro_que: ProQue,
}

impl Forward for Pool2dLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for Pool"));
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

        // Calculate output dimensions
        let output_height = (input_height + 2 * self.lconf.padding[0] - self.lconf.kernel_size[0])
            / self.lconf.stride[0]
            + 1;
        let output_width = (input_width + 2 * self.lconf.padding[1] - self.lconf.kernel_size[1])
            / self.lconf.stride[1]
            + 1;

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

        // Determine pool type
        let pool_type = match self.lconf.pool_type {
            PoolType::Max => 0,
            PoolType::Avg => 1,
        };

        // Build and execute kernel
        let kernel = self
            .pro_que
            .kernel_builder("pool")
            .global_work_size(batch_size * channels * output_height * output_width)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(batch_size as i32)
            .arg(channels as i32)
            .arg(input_height as i32)
            .arg(input_width as i32)
            .arg(self.lconf.kernel_size[0] as i32)
            .arg(self.lconf.kernel_size[1] as i32)
            .arg(self.lconf.stride[0] as i32)
            .arg(self.lconf.stride[1] as i32)
            .arg(self.lconf.padding[0] as i32)
            .arg(self.lconf.padding[1] as i32)
            .arg(output_height as i32)
            .arg(output_width as i32)
            .arg(pool_type)
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

impl ToLayer for conf::Pool2dConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;

        Ok(Box::new(Pool2dLayer {
            lconf,
            pro_que: ProQue::builder()
                .dims(256)
                .src(include_str!("./pool2d.cl"))
                .build()
                .unwrap(),
        }))
    }
}
