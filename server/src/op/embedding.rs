use super::conf::{EmbeddingConf, ToLayer};
use super::{conf, dtype::TensorValue, layer::Forward};
use anyhow::{Result, anyhow};
use ndarray::Axis;
use ndarray::{Array2, ArrayD, s};

pub struct EmbeddingLayer {
    pub lconf: conf::EmbeddingConf,
}

impl Forward for EmbeddingLayer {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        // 获取输入张量（必须是Int64类型）
        let indices = if let TensorValue::Int64(indices_arr) = &input[0] {
            indices_arr
        } else {
            return Err(anyhow!("Embedding input must be Int64 tensor"));
        };

        // 获取权重张量（必须是Float32类型）
        let weight_arr = if let TensorValue::Float32(weight) = &self.lconf.weight {
            weight
        } else {
            return Err(anyhow!("Embedding weight must be Float32 tensor"));
        };

        let weight_shape = weight_arr.shape();
        let embedding_dim = weight_shape[1]; // 例如 768
        let max_index = weight_shape[0] as i64;

        // 索引范围检查
        if indices.iter().any(|&idx| idx < 0 || idx >= max_index) {
            return Err(anyhow!("Index out of embedding table bounds"));
        }

        // 构造新数组：行数 = 索引数量，列数 = embedding_dim
        let mut embedded = Array2::<f32>::zeros((indices.len(), embedding_dim));

        for (i, &idx) in indices.iter().enumerate() {
            let row = weight_arr.index_axis(Axis(0), idx as usize);
            embedded.slice_mut(s![i, ..]).assign(&row);
        }

        // 如果输入是多维的，比如 (B, L)，我们需要 reshape 输出成 (B, L, D)
        let mut output_shape = indices.shape().to_vec();
        output_shape.push(embedding_dim);
        let embedded_reshaped: ArrayD<f32> = embedded.into_shape_clone(output_shape)?;

        Ok(vec![TensorValue::Float32(embedded_reshaped)])
    }
}

impl ToLayer for EmbeddingConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(EmbeddingLayer { lconf: self }))
    }
}
