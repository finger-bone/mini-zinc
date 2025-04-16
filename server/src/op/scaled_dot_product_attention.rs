// Scaled Dot Product Attention implementation (Rust side)
// 模仿BatchNorm/ReLU结构，后续将实现Forward trait并调用OpenCL核

use crate::op::conf::ScaledDotProductAttentionConf;
use crate::op::dtype::TensorValue;
use crate::op::layer::Forward;
use anyhow::{Ok, Result};
use ocl::ProQue;

use super::conf::ToLayer;

pub struct ScaledDotProductAttention {
    pub conf: ScaledDotProductAttentionConf,
    pub pro_que: ProQue,
}

impl ScaledDotProductAttention {
    pub fn new(conf: ScaledDotProductAttentionConf) -> Self {
        // let cl_source = format!(
        //     "#define MAX_SEQ_LEN {}\n{}",
        //     conf.max_seq_len,
        //     include_str!("./scaled_dot_product_attention.cl")
        // );
        let cl_source = include_str!("./scaled_dot_product_attention.cl");
        Self {
            conf,
            pro_que: ProQue::builder().dims(512).src(cl_source).build().unwrap(),
        }
    }
}

impl Forward for ScaledDotProductAttention {
    fn forward(&self, input: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        
        // 处理输入：Q, K, V, 可选的mask
        let TensorValue::Float32(q) = &input[0] else {
            return Err(anyhow::anyhow!(
                "Unsupported input type for Q in ScaledDotProductAttention"
            ));
        };

        let TensorValue::Float32(k) = &input[1] else {
            return Err(anyhow::anyhow!(
                "Unsupported input type for K in ScaledDotProductAttention"
            ));
        };

        let TensorValue::Float32(v) = &input[2] else {
            return Err(anyhow::anyhow!(
                "Unsupported input type for V in ScaledDotProductAttention"
            ));
        };

        let TensorValue::Float32(mask) = &input[3] else {
            return Err(anyhow::anyhow!(
                "Unsupported input type for mask in ScaledDotProductAttention"
            ));
        };

        // 获取维度信息 [batch, heads, seq_len, embed_dim]
        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "Q should have 4 dimensions [batch, heads, seq_len, embed_dim]"
            ));
        }

        let k_shape = k.shape();
        if k_shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "K should have 4 dimensions [batch, heads, seq_len, embed_dim]"
            ));
        }
        let v_shape = v.shape();
        if v_shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "V should have 4 dimensions [batch, heads, seq_len, embed_dim]"
            ));
        }

        let mask_shape = mask.shape();
        if mask_shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "mask should have 4 dimensions [batch, heads, seq_len, seq_len]"
            ));
        }
        let batch = q_shape[0];
        let heads = q_shape[1];
        let seq_len = q_shape[2];
        let embed_dim = q_shape[3];

        // 构建输出张量
        let mut output = vec![0.0; batch * heads * seq_len * embed_dim];
        // 构建OpenCL缓冲区
        let q_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(q.len())
            .copy_host_slice(q.as_slice().unwrap())
            .build()
            .unwrap();
        let k_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(k.len())
            .copy_host_slice(k.as_slice().unwrap())
            .build()
            .unwrap();
        let v_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(v.len())
            .copy_host_slice(v.as_slice().unwrap())
            .build()
            .unwrap();
        let mask_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(mask.len())
            .copy_host_slice(mask.as_slice().unwrap())
            .build()
            .unwrap();
        // 创建临时缓冲区用于存储logits
        let temp_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(batch * heads * seq_len * seq_len)
            .build()
            .unwrap();

        let output_buffer = self
            .pro_que
            .buffer_builder::<f32>()
            .len(output.len())
            .build()
            .unwrap();
        // 构建OpenCL内核
        let kernel = self
            .pro_que
            .kernel_builder("scaled_dot_product_attention")
            .arg(&q_buffer)
            .arg(&k_buffer)
            .arg(&v_buffer)
            .arg(&mask_buffer)
            .arg(&output_buffer)
            .arg(&temp_buffer)
            .arg(batch as i32)
            .arg(heads as i32)
            .arg(seq_len as i32)
            .arg(embed_dim as i32)
            .arg(self.conf.dropout as f32)
            .arg(self.conf.scale.unwrap_or(1.0 / (embed_dim as f32).sqrt()) as f32)
            .build()
            .unwrap();
        // 执行OpenCL内核
        unsafe {
            kernel.enq().unwrap();
        }
        // 读取输出缓冲区
        output_buffer.read(output.as_mut_slice()).enq().unwrap();
        // 返回输出张量
        Ok(vec![TensorValue::Float32(
            ndarray::ArrayD::from_shape_vec(vec![batch, heads, seq_len, embed_dim], output)
                .unwrap(),
        )])
    }
}

impl ToLayer for ScaledDotProductAttentionConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(ScaledDotProductAttention::new(self)))
    }
}
