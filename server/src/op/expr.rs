use anyhow::Result;
use ndarray::ArrayD;
use std::str::FromStr;

use super::{
    conf::{ExprConf, ToLayer},
    layer::Forward,
};

use crate::op::dtype::TensorValue;

use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::map_res,
    multi::separated_list0,
    sequence::delimited,
};

#[derive(Debug, Clone)]
pub enum ExprOp {
    Add,
    Mul,
    Sub,
    Neg,
    Input(usize),
    Constant(f32),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub op: ExprOp,
    pub children: Vec<Expr>,
}

impl Expr {
    pub fn parse(input: &str) -> Result<Self> {
        match expr_parser.parse(input) {
            Ok((_, expr)) => Ok(expr),
            Err(e) => Err(anyhow::anyhow!("Failed to parse expression: {:?}", e)),
        }
    }
}

// --- Parser section ---

fn input_parser(input: &str) -> IResult<&str, Expr> {
    let (input, _) = char('@').parse(input)?;
    let (input, idx) = map_res(digit1, FromStr::from_str).parse(input)?;
    Ok((
        input,
        Expr {
            op: ExprOp::Input(idx),
            children: vec![],
        },
    ))
}

fn constant_parser(input: &str) -> IResult<&str, Expr> {
    let (input, value) =
        map_res(nom::number::complete::recognize_float, FromStr::from_str).parse(input)?;
    Ok((
        input,
        Expr {
            op: ExprOp::Constant(value),
            children: vec![],
        },
    ))
}

fn add_parser(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("add").parse(input)?;
    let (input, children) = delimited(
        char('('),
        separated_list0(char(','), expr_parser),
        char(')'),
    )
    .parse(input)?;
    Ok((
        input,
        Expr {
            op: ExprOp::Add,
            children,
        },
    ))
}

fn mul_parser(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("mul").parse(input)?;
    let (input, children) = delimited(
        char('('),
        separated_list0(char(','), expr_parser),
        char(')'),
    )
    .parse(input)?;
    Ok((
        input,
        Expr {
            op: ExprOp::Mul,
            children,
        },
    ))
}

fn sub_parser(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("sub").parse(input)?;
    let (input, children) = delimited(
        char('('),
        separated_list0(char(','), expr_parser),
        char(')'),
    )
    .parse(input)?;
    Ok((
        input,
        Expr {
            op: ExprOp::Sub,
            children,
        },
    ))
}

fn neg_parser(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("neg").parse(input)?;
    let (input, child) = delimited(char('('), expr_parser, char(')')).parse(input)?;
    Ok((
        input,
        Expr {
            op: ExprOp::Neg,
            children: vec![child],
        }
    ))
}

// 重点：nom 8 写法，使用 `.parse(input)`
fn expr_parser(input: &str) -> IResult<&str, Expr> {
    alt((
        input_parser,
        constant_parser,
        add_parser,
        mul_parser,
        sub_parser,
        neg_parser,
    ))
    .parse(input)
}

// --- Layer implementation ---

pub struct ExprLayer {
    pub ast: Expr,
}

impl Forward for ExprLayer {
    fn forward(&mut self, inputs: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let inputs = inputs
            .iter()
            .map(|v| match v {
                TensorValue::Float32(arr) => arr.clone(),
                TensorValue::Int64(arr) => arr.clone().mapv(|v| v as f32),
                _ => panic!("Unsupported input type for Expr"),
            })
            .collect::<Vec<_>>();
        fn eval_expr(expr: &Expr, inputs: &Vec<ArrayD<f32>>) -> ArrayD<f32> {
            match &expr.op {
                ExprOp::Input(idx) => inputs[*idx].clone(),
                ExprOp::Constant(value) => ArrayD::from_elem(inputs[0].shape(), *value),
                ExprOp::Add => {
                    let mut result = eval_expr(&expr.children[0], inputs);
                    for child in &expr.children[1..] {
                        result += &eval_expr(child, inputs);
                    }
                    result
                }
                ExprOp::Mul => {
                    let mut result = eval_expr(&expr.children[0], inputs);
                    for child in &expr.children[1..] {
                        result *= &eval_expr(child, inputs);
                    }
                    result
                }
                ExprOp::Sub => {
                    let mut result = eval_expr(&expr.children[0], inputs);
                    for child in &expr.children[1..] {
                        result -= &eval_expr(child, inputs);
                    }
                    result
                }
                ExprOp::Neg => {
                    let child = &expr.children[0];
                    -eval_expr(child, inputs)
                }
            }
        }

        Ok(vec![TensorValue::Float32(eval_expr(&self.ast, &inputs))])
    }
}

impl ToLayer for ExprConf {
    fn to_layer(self: Self) -> Result<Box<dyn Forward>> {
        let ast = Expr::parse(&self.expr)?;
        Ok(Box::new(ExprLayer { ast }))
    }
}
