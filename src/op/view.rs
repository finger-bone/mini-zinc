use super::{conf::{self, FromZOpConf}, layer::Forward};
use anyhow::{Ok, Result};
use ndarray::ArrayD;

pub struct ViewLayer {
    pub lconf: conf::ViewConf,
}

impl Forward for ViewLayer {
    fn forward(&self, input: &Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        // Only process the first element
        let input = &input[0];
        
        // Check if the total number of elements matches
        let input_size: usize = self.lconf.input_shape.iter().product();
        let output_size: usize = self.lconf.output_shape.iter().product();
        assert_eq!(input_size, output_size, "Input and output shapes must have the same number of elements");
        
        // Reshape the input tensor to the desired output shape
        let output = input.clone().into_shape_with_order(ndarray::IxDyn(&self.lconf.output_shape)).unwrap();
        vec![output]
    }
}

impl FromZOpConf for conf::ViewConf {
    fn from_zopconf(zopconf: conf::ZOpConf) -> Result<Box<dyn Forward>> {
        let conf::ZOpConf::View(lconf) = zopconf else {
            return Err(anyhow::anyhow!("not View"));
        };
        Ok(Box::new(ViewLayer { lconf }))
    }
}