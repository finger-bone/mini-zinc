use anyhow::Result;
use ndarray::Array1;

use crate::op::dtype::TensorValue;
use crate::op::expr::{Expr, ExprLayer};
use crate::op::layer::Forward;

#[test]
fn test_expr_parse() -> Result<()> {
    // Test simple input
    let expr = Expr::parse("@0")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Input(0)));
    assert!(expr.children.is_empty());

    // Test constant
    let expr = Expr::parse("1.000000")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Constant(1.0)));
    assert!(expr.children.is_empty());

    // Test add operation
    let expr = Expr::parse("add(@0,@1)")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Add));
    assert_eq!(expr.children.len(), 2);

    // Test mul operation
    let expr = Expr::parse("mul(@0,@1)")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Mul));
    assert_eq!(expr.children.len(), 2);

    // Test sub operation
    let expr = Expr::parse("sub(@0,@1)")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Sub));
    assert_eq!(expr.children.len(), 2);

    // Test nested expression
    let expr = Expr::parse("add(mul(@0,@1),add(@2,@3))")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Add));
    assert_eq!(expr.children.len(), 2);

    // Test expression with constant
    let expr = Expr::parse("sub(1.000000,@0)")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Sub));
    assert_eq!(expr.children.len(), 2);
    if let crate::op::expr::ExprOp::Constant(val) = expr.children[0].op {
        assert_eq!(val, 1.0);
    } else {
        panic!("Expected constant");
    }

    Ok(())
}

#[test]
fn test_expr_parse_error() {
    // Test invalid input format
    assert!(Expr::parse("@").is_err());
    assert!(Expr::parse("@a").is_err());

    // Test invalid syntax
    assert!(Expr::parse("add(@0,@1").is_err()); // Missing closing parenthesis
    assert!(Expr::parse("add(@0,,@1)").is_err()); // Empty argument
    assert!(Expr::parse("sub(@0,@1").is_err()); // Missing closing parenthesis
}

#[test]
fn test_expr_layer_forward() -> Result<()> {
    // Test add operation
    let layer = ExprLayer {
        ast: Expr::parse("add(@0,@1)")?,
    };
    let inputs = vec![
        TensorValue::Float32(Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn()),
        TensorValue::Float32(Array1::from_vec(vec![4.0, 5.0, 6.0]).into_dyn()),
    ];
    let output = layer.forward(&inputs).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[3]);
        assert_eq!(output, Array1::from_vec(vec![5.0, 7.0, 9.0]).into_dyn());
    } else {
        panic!("Unexpected output type");
    }

    // Test mul operation
    let layer = ExprLayer {
        ast: Expr::parse("mul(@0,@1)")?,
    };
    let output = layer.forward(&inputs).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[3]);
        assert_eq!(output, Array1::from_vec(vec![4.0, 10.0, 18.0]).into_dyn());
    } else {
        panic!("Unexpected output type");
    }

    // Test sub operation
    let layer = ExprLayer {
        ast: Expr::parse("sub(@0,@1)")?,
    };
    let output = layer.forward(&inputs).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[3]);
        assert_eq!(output, Array1::from_vec(vec![-3.0, -3.0, -3.0]).into_dyn());
    } else {
        panic!("Unexpected output type");
    }

    // Test constant operation
    let layer = ExprLayer {
        ast: Expr::parse("sub(1.000000,@0)")?,
    };
    let output = layer.forward(&inputs).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[3]);
        assert_eq!(output, Array1::from_vec(vec![0.0, -1.0, -2.0]).into_dyn());
    } else {
        panic!("Unexpected output type");
    }

    // Test complex expression
    let layer = ExprLayer {
        ast: Expr::parse("add(mul(@0,@1),add(@0,@1))")?,
    };
    let output = layer.forward(&inputs).unwrap();
    if let TensorValue::Float32(output) = &output[0] {
        assert_eq!(output.shape(), &[3]);
        assert_eq!(output, Array1::from_vec(vec![9.0, 17.0, 27.0]).into_dyn());
    } else {
        panic!("Unexpected output type");
    }

    Ok(())
}
