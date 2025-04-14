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

use super::pnnx_weight_reader::PNNXBinDataType;

pub fn parse_shape_and_dtype(src: &str) -> IResult<&str, (Vec<usize>, PNNXBinDataType)> {
    let parse_shape = delimited(tag("("), separated_list0(tag(","), usize), tag(")"));

    let parse_dtype = map_res(alphanumeric1, |s: &str| match s {
        "f32" => Ok(PNNXBinDataType::Float32),
        "f16" => Ok(PNNXBinDataType::Float16),
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
