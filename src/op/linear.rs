use super::{
    conf::{self, FromZOpConf},
    layer::Forward,
};
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct LinearLayer {
    pub lconf: conf::LinearConf,
    pub pro_que: ProQue,
}

impl Forward for LinearLayer {
    fn forward(&self, input: &Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        // Only process the first element
        let input = &input[0];
        let input_shape = input.shape();
        let input_rank = input_shape.len();

        // Get the feature dimension (last dimension)
        let in_features = input_shape[input_rank - 1];
        assert_eq!(
            in_features, self.lconf.in_features,
            "Input features dimension must match layer configuration"
        );

        // Calculate batch size (product of all dimensions except the last one)
        let batch_size: usize = input_shape[..input_rank - 1].iter().product();

        // Reshape input to 2D: (batch_size, in_features)
        let flattened_input = if input_rank > 2 {
            let flat_shape = vec![batch_size, in_features];
            let mut flattened = ArrayD::zeros(ndarray::IxDyn(&flat_shape));
            let flat_slice = flattened.as_slice_mut().unwrap();
            let input_slice = input.as_slice().unwrap();

            for b in 0..batch_size {
                for f in 0..in_features {
                    flat_slice[b * in_features + f] = input_slice[b * in_features + f];
                }
            }
            flattened
        } else {
            input.clone()
        };

        // Create output buffer
        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(batch_size * self.lconf.out_features)
            .build()
            .unwrap();

        // Create input buffer
        let input_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(flattened_input.len())
            .copy_host_slice(flattened_input.as_slice().unwrap())
            .build()
            .unwrap();

        // Create weights buffer
        let weights_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(self.lconf.weights.len())
            .copy_host_slice(self.lconf.weights.as_slice().unwrap())
            .build()
            .unwrap();

        // Create bias buffer
        let bias_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(self.lconf.bias.len())
            .copy_host_slice(self.lconf.bias.as_slice().unwrap())
            .build()
            .unwrap();

        // Build and execute kernel
        let kernel = self
            .pro_que
            .kernel_builder("linear")
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(&weights_buffer)
            .arg(&bias_buffer)
            .arg(batch_size as i32)
            .arg(self.lconf.in_features as i32)
            .arg(self.lconf.out_features as i32)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        // Read the result from buffer
        let mut flat_output = ArrayD::zeros(ndarray::IxDyn(&[batch_size, self.lconf.out_features]));
        output_buffer
            .read(flat_output.as_slice_mut().unwrap())
            .enq()
            .unwrap();

        // Reshape output to match input dimensions, replacing the last dimension with out_features
        let mut output_shape = input_shape.to_vec();
        output_shape[input_rank - 1] = self.lconf.out_features;
        let mut output = ArrayD::zeros(ndarray::IxDyn(&output_shape));
        let output_slice = output.as_slice_mut().unwrap();
        let flat_output_slice = flat_output.as_slice().unwrap();

        for i in 0..output_slice.len() {
            output_slice[i] = flat_output_slice[i];
        }
        vec![output]
    }
}

impl FromZOpConf for conf::LinearConf {
    fn from_zopconf(zopconf: conf::ZOpConf) -> Result<Box<dyn Forward>> {
        let conf::ZOpConf::Linear(lconf) = zopconf else {
            return Err(anyhow::anyhow!("not Linear"));
        };

        Ok(Box::new(LinearLayer {
            lconf,
            pro_que: ProQue::builder()
                .src(include_str!("./linear.cl"))
                .dims(256)
                .build()
                .unwrap(),
        }))
    }
}
