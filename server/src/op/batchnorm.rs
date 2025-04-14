use super::{
    conf::{self, ToLayer},
    layer::{Forward, TensorValue},
};
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct BatchNormLayer {
    pub lconf: conf::BatchNormConf,
    pub pro_que: ProQue,
    pub gamma: ArrayD<f32>,
    pub beta: ArrayD<f32>,
    pub running_mean: ArrayD<f32>,
    pub running_var: ArrayD<f32>,
}

impl Forward for BatchNormLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for BatchNorm"));
        };
        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size = input.len() / (batch_size * channels);

        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(input.len())
            .build()
            .unwrap();
        let input_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(input.len())
            .copy_host_slice(input.as_slice().unwrap())
            .build()
            .unwrap();

        let mean_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(channels)
            .copy_host_slice(self.running_mean.as_slice().unwrap())
            .build()
            .unwrap();

        let var_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(channels)
            .copy_host_slice(self.running_var.as_slice().unwrap())
            .build()
            .unwrap();

        let gamma_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(channels)
            .copy_host_slice(self.gamma.as_slice().unwrap())
            .build()
            .unwrap();

        let beta_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(channels)
            .copy_host_slice(self.beta.as_slice().unwrap())
            .build()
            .unwrap();

        let kernel = self
            .pro_que
            .kernel_builder("batchnorm")
            .global_work_size(input.len())
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(&mean_buffer)
            .arg(&var_buffer)
            .arg(&gamma_buffer)
            .arg(&beta_buffer)
            .arg(self.lconf.epsilon)
            .arg(channels as i32)
            .arg(spatial_size as i32)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        let mut output = ArrayD::zeros(input.raw_dim());
        output_buffer
            .read(output.as_slice_mut().unwrap())
            .enq()
            .unwrap();
        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for conf::BatchNormConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let lconf = self;
        // Extract channels from the configuration
        let channels = lconf.num_features;
        let gamma = ArrayD::ones(ndarray::IxDyn(&[channels]));
        let beta: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> =
            ArrayD::zeros(ndarray::IxDyn(&[channels]));
        let running_mean = ArrayD::zeros(ndarray::IxDyn(&[channels]));
        let running_var = ArrayD::ones(ndarray::IxDyn(&[channels]));

        Ok(Box::new(BatchNormLayer {
            lconf,
            pro_que: ProQue::builder()
                .dims(512)
                .src(include_str!("./batchnorm.cl"))
                .build()
                .unwrap(),
            gamma,
            beta,
            running_mean,
            running_var,
        }))
    }
}
