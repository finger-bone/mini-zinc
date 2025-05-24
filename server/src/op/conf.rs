use anyhow::Result;

use super::{dtype::DataType, layer::Forward};
use crate::op::dtype::TensorValue;

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

pub struct LinearWithWeightsInputConf {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: TensorValue,
    pub bias: TensorValue,
}

pub struct ViewConf {
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

pub struct LayerNormConf {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
    pub weight: TensorValue,
    pub bias: TensorValue,
}

pub struct RSMNormConf {
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
    pub elementwise_affine: bool,
    pub weight: TensorValue,
    pub bias: TensorValue,
}

pub struct GeLUConf {}

pub struct SiLUConf {}

pub struct TransposeConf {
    pub dim0: isize,
    pub dim1: isize,
}

pub struct ExpandConf {
    pub shape: Vec<usize>,
}

pub struct MaskedFillConf {
    pub value: f32,
}

// Tensor.to                Tensor.to_18             1 1 10 11 copy=False dtype=torch.bool $input=10 #10=(1,1,482,482)f32 #11=(1,1,482,482)bool
pub struct TensorToConf {
    pub target_dtype: DataType,
}

pub struct EmbeddingConf {
    pub weight: TensorValue,
}

// pub struct scaled_dot_product_attention
pub struct ScaledDotProductAttentionConf {
    pub dropout: f32,
    /// NOT YET IMPLEMENTED is_causal: bool
    pub is_causal: bool,
    pub max_seq_len: usize,
    pub scale: Option<f32>,
}

pub trait ToLayer {
    fn to_layer(self: Self) -> Result<Box<dyn Forward>>;
}

pub struct UnsqueezeConf {
    pub axes: Vec<usize>,
}

pub struct CatConf {
    pub dim: isize,
}
