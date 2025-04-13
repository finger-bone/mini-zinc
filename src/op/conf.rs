use anyhow::Result;

use super::layer::{Forward, TensorValue};

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

pub struct Conv2dConf {
    pub kernel_size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub dilation: Vec<usize>,
    pub groups: usize,
    pub filters: usize,
    pub weights: TensorValue,
    pub bias: TensorValue,
}

pub struct Pool2dConf {
    pub kernel_size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub pool_type: PoolType,
}

pub struct AdaptivePool2dConf {
    pub output_size: Vec<usize>,
    pub pool_type: PoolType,
}

pub enum PoolType {
    Max,
    Avg,
}

pub struct LinearConf {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: TensorValue,
    pub bias: TensorValue,
}

pub struct ViewConf {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ExprConf {
    pub expr: String,
}

pub struct FlattenConf {
    pub start_dim: isize,
    pub end_dim: isize,
}

pub trait ToLayer {
    fn to_layer(self: Self) -> Result<Box<dyn Forward>>;
}
