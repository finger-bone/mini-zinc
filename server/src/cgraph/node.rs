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
            AdaptivePool2dConf, Conv2dConf, ExprConf, FlattenConf, GeLUConf, LayerNormConf,
            LinearConf, Pool2dConf, PoolType, ReLUConf,
        },
        layer::{Forward, TensorValue},
    },
};

use crate::op::conf::ToLayer;

use super::pnnx_value_parser::{parse_bool, parse_f32, parse_isize, parse_usize_tuple};

pub enum CGNodeOp {
    Input,
    Output,
    Op(Box<dyn Forward>),
}

impl Debug for CGNodeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CGNodeOp::Input => write!(f, "Input"),
            CGNodeOp::Output => write!(f, "Output"),
            CGNodeOp::Op(_) => write!(f, "Op(Unknown)"),
        }
    }
}

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
                Ok(CGNodeOp::Op(
                    LinearConf {
                        in_features,
                        out_features,
                        weights: weights.get(&line.get_tensor_key("weight")).unwrap().clone(),
                        bias: weights.get(&line.get_tensor_key("bias")).unwrap().clone(),
                    }
                    .to_layer()
                    .unwrap(),
                ))
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
            any => Err(anyhow!("Unsupported operator type {}", any)),
        }
        .unwrap();

        let node = CGNode {
            name: line.op_name.clone(),
            op,
            inputs: line.input_idx_list.clone(),
            outputs: line.output_idx_list.clone(),
        };
        Ok(node)
    }
}
