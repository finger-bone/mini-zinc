use super::{
    conf::{self, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct LinearLayerWithWeightsInput {
    pub lconf: conf::LinearWithWeightsInputConf,
    pub pro_que: ProQue,
}

impl Forward for LinearLayerWithWeightsInput {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // ?
        let TensorValue::Float32(weights) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for Linear"));
        };
        let TensorValue::Float32(input) = &input[1] else {
            return Err(anyhow::anyhow!("Unsupported input type for Linear"));
        };
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
        // let weights = match &self.lconf.weights {
        //     TensorValue::Float32(weights) => {
        //         assert_eq!(
        //             weights.shape(),
        //             &[self.lconf.out_features, self.lconf.in_features]
        //         );
        //         weights
        //     }
        //     _ => return Err(anyhow::anyhow!("Unsupported weights type for Linear")),
        // };
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
            _ => return Err(anyhow::anyhow!("Unsupported bias type for Linear")),
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
            .kernel_builder("linear")
            .global_work_size([batch_size * self.lconf.out_features as usize])
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
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for conf::LinearWithWeightsInputConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf: conf::LinearWithWeightsInputConf = self;
        Ok(Box::new(LinearLayerWithWeightsInput {
            lconf,
            pro_que: ProQue::builder()
                .dims(512)
                .src(include_str!("./linear_naive.cl"))
                .build()
                .unwrap(),
        }))
    }
}
