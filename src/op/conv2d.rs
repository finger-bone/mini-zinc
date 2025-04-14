use super::{
    conf::{self, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct Conv2dLayer {
    pub lconf: conf::Conv2dConf,
    pub pro_que: ProQue,
}

impl Forward for Conv2dLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for Conv2d"));
        };
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = if input_shape.len() > 3 {
            input_shape[3]
        } else {
            1
        };

        // Calculate output dimensions with dilation
        let dilated_kernel_h = self.lconf.kernel_size[0]
            + (self.lconf.kernel_size[0] - 1) * (self.lconf.dilation[0] - 1);
        let dilated_kernel_w = self.lconf.kernel_size[1]
            + (self.lconf.kernel_size[1] - 1) * (self.lconf.dilation[1] - 1);
        let output_height = (input_height + 2 * self.lconf.padding[0] - dilated_kernel_h)
            / self.lconf.stride[0]
            + 1;
        let output_width =
            (input_width + 2 * self.lconf.padding[1] - dilated_kernel_w) / self.lconf.stride[1] + 1;

        // Create output buffer
        let output_size = batch_size * self.lconf.filters * output_height * output_width;
        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(output_size)
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

        let weights = match &self.lconf.weights {
            TensorValue::Float32(weights) => weights,
            _ => panic!("Unsupported weights type for Conv2d"),
        };

        // Create weights buffer
        let weights_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(weights.len())
            .copy_host_slice(weights.as_slice().unwrap())
            .build()
            .unwrap();

        let bias = match &self.lconf.bias {
            TensorValue::Float32(bias) => bias,
            _ => panic!("Unsupported bias type for Conv2d"),
        };

        // Create bias buffer
        let bias_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(bias.len())
            .copy_host_slice(bias.as_slice().unwrap())
            .build()
            .unwrap();

        // Build and execute kernel
        let kernel = self
            .pro_que
            .kernel_builder("conv2d")
            .global_work_size(output_size)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(&weights_buffer)
            .arg(&bias_buffer)
            .arg(batch_size as i32)
            .arg(input_channels as i32)
            .arg(input_height as i32)
            .arg(input_width as i32)
            .arg(self.lconf.filters as i32)
            .arg(self.lconf.kernel_size[0] as i32)
            .arg(self.lconf.kernel_size[1] as i32)
            .arg(self.lconf.stride[0] as i32)
            .arg(self.lconf.stride[1] as i32)
            .arg(self.lconf.padding[0] as i32)
            .arg(self.lconf.padding[1] as i32)
            .arg(self.lconf.dilation[0] as i32)
            .arg(self.lconf.dilation[1] as i32)
            .arg(self.lconf.groups as i32)
            .arg(output_height as i32)
            .arg(output_width as i32)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        // Create output array and read from buffer
        let mut output_shape = vec![batch_size, self.lconf.filters, output_height];
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

impl ToLayer for conf::Conv2dConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;

        Ok(Box::new(Conv2dLayer {
            lconf,
            pro_que: ProQue::builder()
                .dims(256)
                .src(include_str!("./conv2d.cl"))
                .build()
                .unwrap(),
        }))
    }
}
