use super::{
    conf::{self, ToLayer},
    layer::Forward,
};
use crate::op::dtype::TensorValue;
use anyhow::{Ok, Result};

pub struct TransposeLayer {
    pub tconf: conf::TransposeConf,
}

impl Forward for TransposeLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // Only process the first element
        let TensorValue::Float32(input) = &input[0] else {
            return Err(anyhow::anyhow!("Unsupported input type for Transpose"));
        };

        let input_shape = input.shape();
        let ndim = input_shape.len();

        // 确保输入张量至少有两个维度
        assert!(
            ndim >= 2,
            "Input tensor must have at least 2 dimensions for transpose"
        );

        // 获取要交换的维度
        // 如果是负数，使用负数转正数
        let dim0 = if self.tconf.dim0 < 0 {
            (self.tconf.dim0 + ndim as isize) as usize
        } else {
            self.tconf.dim0 as usize
        };
        let dim1 = if self.tconf.dim1 < 0 {
            (self.tconf.dim1 + ndim as isize) as usize
        } else {
            self.tconf.dim1 as usize
        };

        // 验证维度有效性
        assert!(
            dim0 < ndim && dim1 < ndim,
            "Transpose dimensions out of range"
        );

        // 执行转置操作
        // 创建一个轴顺序数组，默认为[0,1,2,...,n-1]
        let mut axes: Vec<usize> = (0..ndim).collect();
        // 交换dim0和dim1的位置
        axes.swap(dim0, dim1);

        // 使用permuted_axes执行转置
        let output = input
            .clone()
            .permuted_axes(axes)
            .as_standard_layout()
            .to_owned();

        output.as_slice().unwrap();

        Ok(vec![TensorValue::Float32(output)])
    }
}

impl ToLayer for conf::TransposeConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        let tconf = self;
        Ok(Box::new(TransposeLayer { tconf }))
    }
}
