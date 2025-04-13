use anyhow::Result;
use ndarray::Array1;

use crate::op::expr::{Expr, ExprLayer};
use crate::op::layer::Forward;

#[test]
fn test_expr_parse() -> Result<()> {
    // Test simple input
    let expr = Expr::parse("@0")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Input(0)));
    assert!(expr.children.is_empty());

    // Test add operation
    let expr = Expr::parse("add(@0,@1)")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Add));
    assert_eq!(expr.children.len(), 2);

    // Test mul operation
    let expr = Expr::parse("mul(@0,@1)")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Mul));
    assert_eq!(expr.children.len(), 2);

    // Test nested expression
    let expr = Expr::parse("add(mul(@0,@1),add(@2,@3))")?;
    assert!(matches!(expr.op, crate::op::expr::ExprOp::Add));
    assert_eq!(expr.children.len(), 2);

    Ok(())
}

#[test]
fn test_expr_parse_error() {
    // Test invalid input format
    assert!(Expr::parse("0").is_err());
    assert!(Expr::parse("@").is_err());
    assert!(Expr::parse("@a").is_err());

    // Test invalid operation
    assert!(Expr::parse("sub(@0,@1)").is_err());

    // Test invalid syntax
    assert!(Expr::parse("add(@0,@1").is_err()); // Missing closing parenthesis
    assert!(Expr::parse("add(@0,,@1)").is_err()); // Empty argument
}

#[test]
fn test_expr_layer_forward() -> Result<()> {
    // Test add operation
    let layer = ExprLayer {
        ast: Expr::parse("add(@0,@1)")?,
    };
    let inputs = vec![
        Array1::from_vec(vec![1.0, 2.0, 3.0]).into_dyn(),
        Array1::from_vec(vec![4.0, 5.0, 6.0]).into_dyn(),
    ];
    let output = layer.forward(&inputs);
    assert_eq!(output[0], Array1::from_vec(vec![5.0, 7.0, 9.0]).into_dyn());

    // Test mul operation
    let layer = ExprLayer {
        ast: Expr::parse("mul(@0,@1)")?,
    };
    let output = layer.forward(&inputs);
    assert_eq!(
        output[0],
        Array1::from_vec(vec![4.0, 10.0, 18.0]).into_dyn()
    );

    // Test complex expression
    let layer = ExprLayer {
        ast: Expr::parse("add(mul(@0,@1),add(@0,@1))")?,
    };
    let output = layer.forward(&inputs);
    assert_eq!(
        output[0],
        Array1::from_vec(vec![9.0, 17.0, 27.0]).into_dyn()
    );

    Ok(())
}
