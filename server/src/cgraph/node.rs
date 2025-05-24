use std::{collections::HashMap, fmt::Debug};

use anyhow::{Ok, Result, anyhow};
use nom::Parser;

use crate::{
    cgraph::{
        pnnx_reader::{self, PNNXKVType},
        pnnx_value_parser::parse_usize,
    },
    op::{
        conf::{
            AdaptivePool2dConf, CatConf, ContiguousConf, Conv2dConf, EmbeddingConf, ExpandConf,
            ExprConf, FlattenConf, GeLUConf, LayerNormConf, LinearConf, LinearWithWeightsInputConf,
            MaskedFillConf, Pool2dConf, PoolType, RSMNormConf, ReLUConf,
            ScaledDotProductAttentionConf, ScalerEqConf, SiLUConf, TensorSplitConf, TensorToConf,
            TransposeConf, UnsqueezeConf, ViewConf,
        },
        layer::Forward,
    },
};

use crate::op::dtype::TensorValue;

use crate::op::conf::ToLayer;

use super::pnnx_value_parser::{
    parse_bool, parse_dtype, parse_f32, parse_isize, parse_usize_tuple,
};

pub enum CGNodeOp {
    Input,
    Output,
    Attribute(TensorValue),
    Op(Box<dyn Forward>),
}

impl Debug for CGNodeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CGNodeOp::Input => write!(f, "Input"),
            CGNodeOp::Output => write!(f, "Output"),
            CGNodeOp::Attribute(_) => write!(f, "Attribute(Value Unknown)"),
            CGNodeOp::Op(_) => write!(f, "Op(Information Unknown)"),
        }
    }
}

#[derive(Debug)]
pub struct CGNode {
    pub name: String,
    pub op: CGNodeOp,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

impl CGNode {
    pub fn from_line(
        line: &pnnx_reader::PNNXLine,
        weights: &HashMap<String, TensorValue>,
    ) -> Result<Self> {
        let op = match line.op_type.as_str() {
            "pnnx.Input" => Ok(CGNodeOp::Input),
            "pnnx.Ouput" => Ok(CGNodeOp::Output),
            // pnnx.Attribute           model.distilbert.embeddings.word_embeddings 0 1 2 @data=(30522,768)f32 #2=(30522,768)f32
            "pnnx.Attribute" => Ok(CGNodeOp::Attribute(
                weights.get(&line.get_tensor_key("data")).unwrap().clone(),
            )),
            "nn.ReLU" => Ok(CGNodeOp::Op(
                ReLUConf { threshold: 0f32 }.to_layer().unwrap(),
            )),
            "nn.Conv2d" => {
                // bias=True dilation=(1,1) groups=1 in_channels=64 kernel_size=(3,3) out_channels=64 padding=(1,1) padding_mode=zeros stride=(1,1)
                let (_, dilation) = parse_usize_tuple
                    .parse(&line.get("dilation", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                let (_, groups) = parse_usize
                    .parse(line.get("groups", PNNXKVType::Attr).unwrap().value.as_str())
                    .unwrap();
                let (_, _in_channels) = parse_usize
                    .parse(
                        line.get("in_channels", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let (_, kernel_size) = parse_usize_tuple
                    .parse(
                        line.get("kernel_size", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let (_, out_channels) = parse_usize
                    .parse(
                        line.get("out_channels", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let (_, padding) = parse_usize_tuple
                    .parse(
                        line.get("padding", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let (_, stride) = parse_usize_tuple
                    .parse(line.get("stride", PNNXKVType::Attr).unwrap().value.as_str())
                    .unwrap();

                Ok(CGNodeOp::Op(
                    Conv2dConf {
                        dilation,
                        kernel_size,
                        stride,
                        padding,
                        groups,
                        filters: out_channels,
                        weights: weights.get(&line.get_tensor_key("weight")).unwrap().clone(),
                        bias: weights.get(&line.get_tensor_key("bias")).unwrap().clone(),
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            "nn.MaxPool2d" => {
                // ceil_mode=False dilation=(1,1) kernel_size=(3,3) padding=(1,1) return_indices=False stride=(2,2)
                let (_, _dilation) = parse_usize_tuple
                    .parse(&line.get("dilation", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                let (_, kernel_size) = parse_usize_tuple
                    .parse(
                        line.get("kernel_size", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let (_, padding) = parse_usize_tuple
                    .parse(
                        line.get("padding", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let (_, stride) = parse_usize_tuple
                    .parse(line.get("stride", PNNXKVType::Attr).unwrap().value.as_str())
                    .unwrap();
                Ok(CGNodeOp::Op(
                    Pool2dConf {
                        kernel_size,
                        padding,
                        stride,
                        pool_type: PoolType::Max,
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            "pnnx.Expression" => {
                // pnnx.Expression          pnnx_expr_14             2 1 6 3 7 expr=add(@0,@1) #6=(1,64,56,56)f32 #3=(1,64,56,56)f32 #7=(1,64,56,56)f32
                let expr = line.get("expr", PNNXKVType::Attr).unwrap().value.clone();
                Ok(CGNodeOp::Op(ExprConf { expr }.to_layer().unwrap()))
            }
            "nn.AdaptiveAvgPool2d" => {
                // nn.AdaptiveAvgPool2d     model.avgpool            1 1 46 47 output_size=(1,1) #46=(1,512,7,7)f32 #47=(1,512,1,1)f32
                let (_, output_size) = parse_usize_tuple
                    .parse(
                        line.get("output_size", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                Ok(CGNodeOp::Op(
                    AdaptivePool2dConf {
                        output_size,
                        pool_type: PoolType::Avg,
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            // torch.flatten            torch.flatten_0          1 1 47 48 end_dim=-1 start_dim=1 $input=47 #47=(1,512,1,1)f32 #48=(1,512)f32
            "torch.flatten" => {
                let (_, end_dim) = parse_isize
                    .parse(&line.get("end_dim", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                let (_, start_dim) = parse_isize
                    .parse(&line.get("start_dim", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(
                    FlattenConf { end_dim, start_dim }.to_layer().unwrap(),
                ))
            }
            "nn.Linear" => {
                // nn.Linear                model.fc                 1 1 48 49 bias=True in_features=512 out_features=1000 @bias=(1000)f32 @weight=(1000,512)f32 #48=(1,512)f32 #49=(1,1000)f32
                let (_, in_features) = parse_usize
                    .parse(&line.get("in_features", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                let (_, out_features) = parse_usize
                    .parse(&line.get("out_features", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                if line.input_idx_list.len() == 1 {
                    // let bias = if weights.get(&line.get_tensor_key("bias")).unwrap().clone()
                    let bias = match weights.get(&line.get_tensor_key("bias")) {
                        Some(tensor) => Some(tensor.clone()),
                        None => None,
                    };
                    Ok(CGNodeOp::Op(
                        LinearConf {
                            in_features,
                            out_features,
                            weights: weights.get(&line.get_tensor_key("weight")).unwrap().clone(),
                            bias: bias,
                        }
                        .to_layer()
                        .unwrap(),
                    ))
                } else {
                    let bias = match weights.get(&line.get_tensor_key("bias")) {
                        Some(tensor) => Some(tensor.clone()),
                        None => None,
                    };

                    Ok(CGNodeOp::Op(
                        LinearWithWeightsInputConf {
                            in_features,
                            out_features,
                            bias: bias,
                        }
                        .to_layer()
                        .unwrap(),
                    ))
                }
            }
            "pnnx.Output" => Ok(CGNodeOp::Output),
            // nn.LayerNorm             model.distilbert.transformer.layer.4.output_layer_norm 1 1 111 112 elementwise_affine=True eps=1.000000e-12 normalized_shape=(768) @bias=(768)f32 @weight=(768)f32 #111=(1,482,768)f32 #112=(1,482,768)f32
            "nn.LayerNorm" => {
                let (_, eps) = parse_f32
                    .parse(&line.get("eps", PNNXKVType::Attr).unwrap().value.as_str())
                    .unwrap();
                let (_, elementwise_affine) = parse_bool
                    .parse(
                        &line
                            .get("elementwise_affine", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let (_, normalized_shape) = parse_usize_tuple
                    .parse(
                        &line
                            .get("normalized_shape", PNNXKVType::Attr)
                            .unwrap()
                            .value
                            .as_str(),
                    )
                    .unwrap();
                let layer_norm_weight =
                    weights.get(&line.get_tensor_key("weight")).unwrap().clone();
                let layer_norm_bias = weights.get(&line.get_tensor_key("bias")).unwrap().clone();

                Ok(CGNodeOp::Op(
                    LayerNormConf {
                        normalized_shape,
                        eps,
                        elementwise_affine,
                        weight: layer_norm_weight,
                        bias: layer_norm_bias,
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            "F.gelu" => Ok(CGNodeOp::Op(GeLUConf {}.to_layer().unwrap())),
            // Tensor.view              Tensor.view_19           1 1 13 14 shape=(1,482,12,64) $input=13 #13=(1,482,768)f32 #14=(1,482,12,64)f32
            "Tensor.view" | "Tensor.reshape" => {
                let (_, output_shape) = parse_usize_tuple
                    .parse(line.get("shape", PNNXKVType::Attr).unwrap().value.as_str())
                    .unwrap();
                Ok(CGNodeOp::Op(ViewConf { output_shape }.to_layer().unwrap()))
            }
            // torch.transpose          torch.transpose_45       1 1 20 21 dim0=1 dim1=2 $input=20 #20=(1,482,12,64)f32 #21=(1,12,482,64)f32
            "torch.transpose" => {
                let (_, dim0) = parse_isize
                    .parse(&line.get("dim0", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                let (_, dim1) = parse_isize
                    .parse(&line.get("dim1", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(
                    TransposeConf { dim0, dim1 }.to_layer().unwrap(),
                ))
            }
            //Tensor.expand            Tensor.expand_16         1 1 7 8 shape=(1,1,482,482) $input=7 #7=(1,1,1,482)i64 #8=(1,1,482,482)i64
            "Tensor.expand" => {
                let (_, output_shape) = parse_usize_tuple
                    .parse(line.get("shape", PNNXKVType::Attr).unwrap().value.as_str())
                    .unwrap();
                Ok(CGNodeOp::Op(
                    ExpandConf {
                        shape: output_shape,
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            // Tensor.masked_fill       Tensor.masked_fill_69    2 1 10 11 12 value=-3.402823e+38 $input=10 $mask=11 #10=(1,1,482,482)f32 #11=(1,1,482,482)bool #12=(1,1,482,482)f32
            "Tensor.masked_fill" => {
                let (_, value) = parse_f32
                    .parse(&line.get("value", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(MaskedFillConf { value }.to_layer().unwrap()))
            }
            "Tensor.to" => {
                // Tensor.to                Tensor.to_17             1 1 8 9 copy=False dtype=torch.float $input=8 #8=(1,1,482,482)i64 #9=(1,1,482,482)f32
                let (_, dtype) = parse_dtype
                    .parse(&line.get("dtype", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(
                    TensorToConf {
                        target_dtype: dtype,
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            // nn.Embedding             pnnx_unique_0            1 1 0 3 embedding_dim=768 num_embeddings=30522 sparse=False @weight=(30522,768)f32 #0=(1,482)i64 #3=(1,482,768)f32
            "nn.Embedding" => {
                let weight = weights.get(&line.get_tensor_key("weight")).unwrap();
                Ok(CGNodeOp::Op(
                    EmbeddingConf {
                        weight: weight.clone(),
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            // F.scaled_dot_product_attention F.scaled_dot_product_attention_1862 4 1 87 88 89 9 90 dropout_p=0.000000e+00 is_causal=False scale=1.250000e-01 $query=87 $key=88 $value=89 $attn_mask=9 #87=(1,9,512,64)f32 #88=(1,9,512,64)f32 #89=(1,9,512,64)f32 #9=(1,1,512,512)f32 #90=(1,9,512,64)f32
            "F.scaled_dot_product_attention" => {
                let scale = match &line.get("scale", PNNXKVType::Attr) {
                    Some(v) => {
                        let (_, scale) = parse_f32.parse(v.value.as_str()).unwrap();
                        Some(scale)
                    }
                    None => None,
                };
                let (_, is_causal) = parse_bool
                    .parse(&line.get("is_causal", PNNXKVType::Attr).unwrap().value)
                    .unwrap();

                // let input_q_shape = &line.get("query", PNNXKVType::Input)
                // let dropout = parse_f32

                let input_q_blob = &line.get("query", PNNXKVType::Input).unwrap().value;
                let input_q_shape = &line.get(&input_q_blob, PNNXKVType::Shape).unwrap().value;
                let (_, input_q_shape) = parse_usize_tuple.parse(input_q_shape.as_str()).unwrap();
                let max_seq_len = input_q_shape[1];

                Ok(CGNodeOp::Op(
                    ScaledDotProductAttentionConf {
                        scale,
                        dropout: 0.0,
                        is_causal,
                        max_seq_len,
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            "torch.eq" => {
                // torch.eq                 torch.eq_460             1 1 7 8 other=0 $input=7 #7=(1,1,32,32)f32 #8=(1,1,32,32)bool
                let (_, other) = parse_f32
                    .parse(&line.get("other", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(ScalerEqConf { other }.to_layer().unwrap()))
            }
            // Tensor.contiguous        Tensor.contiguous_523    1 1 28 40 memory_format=torch.contiguous_format $input=28 #28=(1,9,32,64)f32 #40=(1,9,32,64)f32
            "Tensor.contiguous" => Ok(CGNodeOp::Op(ContiguousConf {}.to_layer().unwrap())),
            // torch.cat                torch.cat_1050           2 1 73 71 74 dim=-1 #73=(1,9,32,32)f32 #71=(1,9,32,32)f32 #74=(1,9,32,64)f32
            "torch.cat" => {
                let (_, dim) = parse_isize
                    .parse(&line.get("dim", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(CatConf { dim }.to_layer().unwrap()))
            }
            // torch.tensor_split       Tensor.slice_1395        1 2 63 76 77 dim=3 indices=(32) #63=(1,3,32,64)f32 #76=(1,3,32,32)f32 #77=(1,3,32,32)f32
            "torch.tensor_split" => {
                let (_, indices) = parse_usize_tuple
                    .parse(&line.get("indices", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                let (_, dim) = parse_isize
                    .parse(&line.get("dim", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(
                    TensorSplitConf { dim, indices }.to_layer().unwrap(),
                ))
            }
            // nn.RMSNorm               rmsnorm_3                1 1 96 97 elementwise_affine=True eps=1.000000e-05 normalized_shape=(576) @weight=(576)f32 $input=96 #96=(1,32,576)f32 #97=(1,32,576)f32
            "nn.RMSNorm" => {
                let (_, eps) = parse_f32
                    .parse(&line.get("eps", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                let (_, elementwise_affine) = parse_bool
                    .parse(
                        &line
                            .get("elementwise_affine", PNNXKVType::Attr)
                            .unwrap()
                            .value,
                    )
                    .unwrap();
                let (_, normalized_shape) = parse_usize_tuple
                    .parse(
                        &line
                            .get("normalized_shape", PNNXKVType::Attr)
                            .unwrap()
                            .value,
                    )
                    .unwrap();
                let layer_norm_weight =
                    weights.get(&line.get_tensor_key("weight")).unwrap().clone();
                Ok(CGNodeOp::Op(
                    RSMNormConf {
                        normalized_shape,
                        eps,
                        elementwise_affine,
                        weight: layer_norm_weight,
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            // torch.unsqueeze          torch.unsqueeze_1255     1 1 221 222 dim=2 $input=221 #221=(1,3,32,64)f32 #222=(1,3,1,32,64)f32
            "torch.unsqueeze" => {
                let (_, dim) = parse_isize
                    .parse(&line.get("dim", PNNXKVType::Attr).unwrap().value)
                    .unwrap();
                Ok(CGNodeOp::Op(
                    UnsqueezeConf { axes: vec![dim] }.to_layer().unwrap(),
                ))
            }
            // nn.SiLU                  model.model.layers.4.mlp.act_fn 1 1 239 240 #239=(1,32,1536)f32 #240=(1,32,1536)f32
            "nn.SiLU" => Ok(CGNodeOp::Op(SiLUConf {}.to_layer().unwrap())),
            any => Err(anyhow!("Unsupported operator type {}", any)),
        }
        .unwrap();
        // eprintln!("{}", line.op_name);
        let node = CGNode {
            name: line.op_name.clone(),
            op,
            inputs: line.input_idx_list.clone(),
            outputs: line.output_idx_list.clone(),
        };
        Ok(node)
    }
}
