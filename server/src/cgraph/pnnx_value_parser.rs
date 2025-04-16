use anyhow::anyhow;
use nom::bytes::complete::take_while;
use nom::character::complete::{isize, usize};
use nom::number::complete::recognize_float;
use nom::{
    IResult, Parser,
    bytes::complete::tag,
    character::complete::alphanumeric1,
    combinator::map_res,
    multi::separated_list0,
    sequence::{delimited, pair},
};

use crate::op::dtype::DataType;

pub fn parse_shape_and_dtype(src: &str) -> IResult<&str, (Vec<usize>, DataType)> {
    let parse_shape = delimited(tag("("), separated_list0(tag(","), usize), tag(")"));

    let parse_dtype = map_res(alphanumeric1, |s: &str| match s {
        "f32" => Ok(DataType::Float32),
        "f16" => Ok(DataType::Float16),
        "bf16" => Ok(DataType::BFloat16),
        "bool" => Ok(DataType::Boolean),
        "i64" => Ok(DataType::Int64),
        _ => Err(format!("Unknown dtype: {}", s)),
    });

    pair(parse_shape, parse_dtype).parse(src)
}

pub fn parse_usize_tuple(src: &str) -> IResult<&str, Vec<usize>> {
    delimited(tag("("), separated_list0(tag(","), usize), tag(")")).parse(src)
}

pub fn parse_usize(src: &str) -> IResult<&str, usize> {
    usize.parse(src)
}

pub fn parse_isize(src: &str) -> IResult<&str, isize> {
    isize.parse(src)
}

pub fn parse_bool(src: &str) -> IResult<&str, bool> {
    let parse_true = tag("True").map(|_| true);
    let parse_false = tag("False").map(|_| false);

    parse_true.or(parse_false).parse(src)
}

pub fn parse_f32(input: &str) -> IResult<&str, f32> {
    map_res(recognize_float, |s: &str| s.parse::<f32>()).parse(input)
}

pub fn parse_dtype(input: &str) -> IResult<&str, DataType> {
    map_res(
        // 使用 take_while 代替 alphanumeric1 以匹配包含点的类型名称
        take_while(|c: char| c.is_ascii_alphanumeric() || c == '.'), 
        |s: &str| match s {
            "torch.float32" | "torch.float"=> Ok(DataType::Float32),
            "torch.float16" => Ok(DataType::Float16),
            "torch.bfloat16" => Ok(DataType::BFloat16),
            "torch.bool" => Ok(DataType::Boolean),
            "torch.int64" => Ok(DataType::Int64),
            _ => Err(anyhow!("Invalid DataType: {}", s)),
        },
    )
    .parse(input)
}