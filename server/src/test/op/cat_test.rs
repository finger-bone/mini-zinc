use crate::op::{conf::{CatConf, ToLayer},dtype::TensorValue};
use anyhow::Result;
use ndarray::ArrayD;

#[test]
fn test_cat_basic() -> Result<()> {
    let conf = CatConf { dim: 0 };
    let mut layer = conf.to_layer()?;

    let input1 = TensorValue::Float32(ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?);
    let input2 = TensorValue::Float32(ArrayD::from_shape_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0])?);

    let output = layer.forward(&vec![input1, input2])?;

    let expected = ArrayD::from_shape_vec(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    match &output[0] {
        TensorValue::Float32(arr) => {
            assert_eq!(arr.shape(), expected.shape());
            assert_eq!(arr.iter().cloned().collect::<Vec<_>>(), expected.iter().cloned().collect::<Vec<_>>());
        },
        _ => panic!("Unexpected output type"),
    }

    Ok(())
}