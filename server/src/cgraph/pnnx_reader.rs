use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use nom::AsChar;
use nom::combinator::opt;
use nom::sequence::separated_pair;
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while1},
    multi::separated_list0,
};

use anyhow::{Result, anyhow};
use nom::character::complete::space1;
use nom::character::complete::usize;

use crate::op::dtype::DataType;

use super::pnnx_value_parser::parse_shape_and_dtype;

pub struct PNNXReaderResult {
    pub num_layers: usize,
    pub num_blobs: usize,
    pub lines: Vec<PNNXLine>,
}

#[derive(Clone, Debug)]
pub struct PNNXLine {
    pub op_type: String,
    pub op_name: String,

    pub input_idx_list: Vec<usize>,
    pub output_idx_list: Vec<usize>,
    pub kvs: Vec<PNNXKV>,
}

impl PNNXLine {
    pub fn get(&self, key: &str, expect_type: PNNXKVType) -> Option<&PNNXKV> {
        for kv in &self.kvs {
            if kv.kv_type == expect_type && kv.key == key {
                return Some(&kv);
            }
        }
        return None;
    }

    pub fn get_tensor_key(&self, key: &str) -> String {
        format!("{}.{}", self.op_name, key)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PNNXKVType {
    Attr,
    Input,
    Shape,
    Tensor,
}

#[derive(Clone, Debug)]
pub struct PNNXKV {
    pub kv_type: PNNXKVType,
    pub key: String,
    pub value: String,
}

impl PNNXReaderResult {
    fn parse_num_layers_and_num_blobs(src: &str) -> IResult<&str, (usize, usize)> {
        separated_pair(usize, space1, usize).parse(src)
    }

    fn parse_non_blank(input: &str) -> IResult<&str, &str> {
        take_while1(|c: char| !c.is_whitespace() && !c.is_newline()).parse(input)
    }

    fn parse_line_op_type_and_op_name(src: &str) -> IResult<&str, (String, String)> {
        let (rem, (op_type, op_name)) =
            separated_pair(Self::parse_non_blank, space1, Self::parse_non_blank).parse(src)?;
        Ok((rem, (op_type.to_string(), op_name.to_string())))
    }

    fn parse_io_idx(src: &str) -> IResult<&str, (Vec<usize>, Vec<usize>)> {
        let (rem, io_idx_list) = separated_list0(space1, usize).parse(src)?;
        let input_cnt = io_idx_list[0];
        let output_cnt = io_idx_list[1];
        let input_idx_list = io_idx_list[2..2 + input_cnt].to_vec();
        let output_idx_list = io_idx_list[2 + input_cnt..2 + input_cnt + output_cnt].to_vec();
        Ok((rem, (input_idx_list, output_idx_list)))
    }

    fn parse_one_kv(src: &str) -> IResult<&str, PNNXKV> {
        let (rem, prefix_opt) = opt(alt((tag("@"), tag("$"), tag("#")))).parse(src)?;

        let kv_type = match prefix_opt {
            Some("@") => PNNXKVType::Tensor,
            Some("$") => PNNXKVType::Input,
            Some("#") => PNNXKVType::Shape,
            None => PNNXKVType::Attr,
            _ => unreachable!(),
        };
        let (rem, (key, value)) = separated_pair(
            take_while1(|c| c != '='),
            tag("="),
            take_while1(|c| c != ' '),
        )
        .parse(rem)?;
        Ok((
            rem,
            PNNXKV {
                kv_type,
                key: key.to_string(),
                value: value.to_string(),
            },
        ))
    }

    fn parse_line(src: &str) -> IResult<&str, PNNXLine> {
        let (rem, (op_type, op_name)) = Self::parse_line_op_type_and_op_name(src)?;
        let (rem, _) = space1.parse(rem)?;
        let (rem, (input_idx_list, output_idx_list)) = Self::parse_io_idx(rem)?;
        if rem.trim().is_empty() {
            return Ok((
                rem,
                PNNXLine {
                    op_type,
                    op_name,
                    input_idx_list,
                    output_idx_list,
                    kvs: Vec::new(),
                },
            ));
        }
        let (rem, _) = space1.parse(rem)?;
        let (rem, kvs) = separated_list0(space1, Self::parse_one_kv).parse(rem)?;
        Ok((
            rem,
            PNNXLine {
                op_type,
                op_name,
                input_idx_list,
                output_idx_list,
                kvs,
            },
        ))
    }

    pub fn from_text(src: &str) -> Result<PNNXReaderResult> {
        let mut lines = src.lines();

        let magic_line: &str = lines
            .next()
            .ok_or_else(|| anyhow!("magic line not found"))?;
        if magic_line != "7767517" {
            return Err(anyhow!("magic line mismatch"));
        }

        let num_line: &str = lines
            .next()
            .ok_or_else(|| anyhow!("num layers and num blobs not found"))?;
        let (_, (num_layers, num_blobs)) = Self::parse_num_layers_and_num_blobs(num_line)
            .map_err(|e| anyhow!("failed to parse layer/blob line: {:?}", e))?;

        let mut pnnx_lines = Vec::new();
        for line in lines {
            let (_, pnnx_line) =
                Self::parse_line(line).map_err(|e| anyhow!("failed to parse line: {:?}", e))?;
            pnnx_lines.push(pnnx_line);
        }

        Ok(Self {
            num_layers,
            num_blobs,
            lines: pnnx_lines,
        })
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<PNNXReaderResult> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut src = String::new();
        reader.read_to_string(&mut src)?;
        Self::from_text(&src)
    }
}

impl PNNXReaderResult {
    pub fn get_shape_and_dtype_map(
        &self,
    ) -> (HashMap<String, Vec<usize>>, HashMap<String, DataType>) {
        let mut shape_map = HashMap::<String, Vec<usize>>::new();
        let mut dtype_map = HashMap::<String, DataType>::new();
        for line in &self.lines {
            for kv in &line.kvs {
                match kv.kv_type {
                    PNNXKVType::Tensor => {
                        let (_, (shape, dtype)) = parse_shape_and_dtype(&kv.value).unwrap();
                        shape_map.insert(line.get_tensor_key(&kv.key), shape);
                        dtype_map.insert(line.get_tensor_key(&kv.key), dtype);
                    }
                    _ => {
                        continue;
                    }
                }
            }
        }
        (shape_map, dtype_map)
    }
}
