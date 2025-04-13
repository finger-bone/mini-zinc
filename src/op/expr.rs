use anyhow::Result;
use ndarray::ArrayD;
use std::str::FromStr;

use super::{
    conf::{FromZOpConf, ZOpConf},
    layer::Forward,
};

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{char, digit1},
    combinator::map_res,
    multi::separated_list0,
    sequence::delimited,
    IResult, Parser,
};

#[derive(Debug, Clone)]
pub enum ExprOp {
    Add,
    Mul,
    Input(usize),
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
    Ok((input, Expr { op: ExprOp::Input(idx), children: vec![] }))
}

fn add_parser(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("add").parse(input)?;
    let (input, children) = delimited(
        char('('),
        separated_list0(char(','), expr_parser),
        char(')'),
    ).parse(input)?;
    Ok((input, Expr { op: ExprOp::Add, children }))
}

fn mul_parser(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("mul").parse(input)?;
    let (input, children) = delimited(
        char('('),
        separated_list0(char(','), expr_parser),
        char(')'),
    ).parse(input)?;
    Ok((input, Expr { op: ExprOp::Mul, children }))
}

// 重点：nom 8 写法，使用 `.parse(input)`
fn expr_parser(input: &str) -> IResult<&str, Expr> {
    alt((
        input_parser,
        add_parser,
        mul_parser,
    )).parse(input)
}

// --- Layer implementation ---

pub struct ExprLayer {
    pub ast: Expr,
}

impl Forward for ExprLayer {
    fn forward(&self, inputs: &Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        fn eval_expr(expr: &Expr, inputs: &Vec<ArrayD<f32>>) -> ArrayD<f32> {
            match &expr.op {
                ExprOp::Input(idx) => inputs[*idx].clone(),
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
            }
        }

        vec![eval_expr(&self.ast, inputs)]
    }
}

impl FromZOpConf for ExprLayer {
    fn from_zopconf(zopconf: ZOpConf) -> Result<Box<dyn Forward>> {
        let lconf = match zopconf {
            ZOpConf::Expression(conf) => conf,
            _ => return Err(anyhow::anyhow!("Expected Expression variant")),
        };

        let ast = Expr::parse(&lconf.expr)?;
        Ok(Box::new(ExprLayer { ast }))
    }
}