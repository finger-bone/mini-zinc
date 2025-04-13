use super::{
    conf::{self, FromZOpConf},
    layer::Forward,
};
use anyhow::{Ok, Result};
use ndarray::ArrayD;
use ocl::ProQue;

pub struct ReLULayer {
    pub lconf: conf::ReLUConf,
    pub pro_que: ProQue,
}

impl Forward for ReLULayer {
    fn forward(&self, input: &Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        // Only process the first element
        let input = &input[0];
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
        let kernel = self
            .pro_que
            .kernel_builder("relu")
            .arg(&input_buffer)
            .arg(&output_buffer)
            .arg(self.lconf.threshold)
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
        vec![output]
    }
}

impl FromZOpConf for conf::ReLUConf {
    fn from_zopconf(zopconf: conf::ZOpConf) -> Result<Box<dyn Forward>> {
        let conf::ZOpConf::ReLU(lconf) = zopconf else {
            return Err(anyhow::anyhow!("not ReLU"));
        };
        Ok(Box::new(ReLULayer {
            lconf,
            pro_que: ProQue::builder()
                .src(include_str!("./relu.cl"))
                .dims(256)
                .build()
                .unwrap(),
        }))
    }
}
