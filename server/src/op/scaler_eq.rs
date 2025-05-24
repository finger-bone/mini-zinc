use super::{
    conf::{ScalerEqConf, ToLayer},
    dtype::TensorValue,
    layer::Forward,
};
use anyhow::Result;
use ndarray::{ArrayD, Zip};

pub struct ScalerEqLayer {
    lconf: ScalerEqConf,
}

impl ToLayer for ScalerEqConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(ScalerEqLayer { lconf: self }))
    }
}

impl Forward for ScalerEqLayer {
    fn forward(&mut self, inputs: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        if let TensorValue::Float32(input) = inputs[0].clone() {
            let other = self.lconf.other;
            let eps = 1e-5f32;

            let mut output = ArrayD::<bool>::default(input.raw_dim()); // 构造形状一致的输出数组

            Zip::from(&mut output)
                .and(&input)
                .par_for_each(|out, &inp| {
                    *out = (inp - other).abs() < eps;
                });

            return Ok(vec![TensorValue::Boolean(output)]);
        }

        Err(anyhow::anyhow!("Unsupported input type for ScaleEq"))
    }
}
