use anyhow::Result;
use ndarray::ArrayD;

use super::layer::Forward;

pub struct ReLUConf {
    pub threshold: f32,
}

pub struct SigmoidConf {}

pub struct TanhConf {}

pub struct SoftmaxConf {
    pub axis: i32,
}

pub struct BatchNormConf {
    pub epsilon: f32,
    pub momentum: f32,
    pub num_features: usize,
}

pub struct ConvConf {
    pub kernel_size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    pub groups: usize,
    pub filters: usize,
    pub weights: ArrayD<f32>,
    pub bias: ArrayD<f32>,
}

pub struct PoolConf {
    pub kernel_size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub pool_type: PoolType,
}

pub enum PoolType {
    Max,
    Avg,
}

pub struct LinearConf {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: ArrayD<f32>,
    pub bias: ArrayD<f32>,
}

pub struct ViewConf {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

pub enum ZOpConf {
    Unkown,
    ReLU(ReLUConf),
    Sigmoid(SigmoidConf),
    Tanh(TanhConf),
    Softmax(SoftmaxConf),
    BatchNorm(BatchNormConf),
    Conv(ConvConf),
    Pool(PoolConf),
    Linear(LinearConf),
    View(ViewConf),
}

pub trait FromZOpConf {
    fn from_zopconf(zopconf: ZOpConf) -> Result<Box<dyn Forward>>;
}
