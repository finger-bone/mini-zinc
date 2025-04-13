use std::{collections::HashMap, path::Path};

use anyhow::{Ok, Result, anyhow};
use nom::Parser;

use crate::{
    cgraph::{
        pnnx_reader::{self, PNNXKVType},
        pnnx_value_parser::parse_usize,
        pnnx_weight_reader::load_pnnx_zip_bin,
    },
    op::{
        conf::{
            AdaptivePool2dConf, Conv2dConf, ExprConf, FlattenConf, LinearConf, Pool2dConf,
            PoolType, ReLUConf,
        },
        layer::{Forward, TensorValue},
    },
};
use nom::character::complete::usize;

use crate::op::conf::ToLayer;

use super::{
    pnnx_reader::PNNXReaderResult,
    pnnx_value_parser::{parse_isize, parse_shape_and_dtype, parse_usize_tuple},
    pnnx_weight_reader,
};

pub enum CGNodeOp {
    Input,
    Output,
    Op(Box<dyn Forward>),
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
                let (_, in_channels) = parse_usize
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
                        weights: weights.get("weight").unwrap().clone(),
                        bias: weights.get("bias").unwrap().clone(),
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            "nn.MaxPool2d" => {
                // ceil_mode=False dilation=(1,1) kernel_size=(3,3) padding=(1,1) return_indices=False stride=(2,2)
                let (_, dilation) = parse_usize_tuple
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
                        weights: weights.get("weight").unwrap().clone(),
                        bias: weights.get("bias").unwrap().clone(),
                    }
                    .to_layer()
                    .unwrap(),
                ))
            }
            _ => Err(anyhow!("Unsupported operator type.")),
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
