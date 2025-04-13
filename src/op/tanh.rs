use super::{
    conf::{self, FromZOpConf},
    layer::Forward,
};
use anyhow::{Ok, Result};
use ocl::ProQue;

pub struct TanhLayer {
    pub pro_que: ProQue,
}

impl Forward for TanhLayer {
    fn forward(&self, input: &Vec<ndarray::ArrayD<f32>>) -> Vec<ndarray::ArrayD<f32>> {
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
            .kernel_builder("tanh")
            .arg(&input_buffer)
            .arg(&output_buffer)
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

impl FromZOpConf for conf::TanhConf {
    fn from_zopconf(zopconf: conf::ZOpConf) -> Result<Box<dyn Forward>> {
        let conf::ZOpConf::Tanh(lconf) = zopconf else {
            return Err(anyhow::anyhow!("not Tanh"));
        };
        Ok(Box::new(TanhLayer {
            pro_que: ProQue::builder()
                .src(include_str!("./tanh.cl"))
                .dims(256)
                .build()
                .unwrap(),
        }))
    }
}
