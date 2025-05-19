use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use warp::Filter;

use crate::cgraph::graph::ComputationGraph;
use crate::op::dtype::TensorValue;

// 服务器状态结构体
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerStatus {
    pub is_busy: bool,
}

// 推理请求结构体
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InferenceRequest {
    pub inputs: HashMap<usize, TensorValueWrapper>,
}

// 推理响应结构体
#[derive(Clone, Debug, Serialize)]
pub struct InferenceResponse {
    pub outputs: HashMap<usize, TensorValueWrapper>,
    pub success: bool,
    pub message: String,
    pub duration_ms: u128, // 新增字段
}

// TensorValue的包装器，用于序列化和反序列化
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorValueWrapper {
    pub dtype: String,
    pub shape: Vec<usize>,
    // 支持多种数据类型
    #[serde(default)]
    pub data_f32: Vec<f32>,
    #[serde(default)]
    pub data_i64: Vec<i64>,
    #[serde(default)]
    pub data_bf16: Vec<f32>, // bf16数据以f32形式传输，转换时处理
    #[serde(default)]
    pub data_f16: Vec<f32>, // f16数据以f32形式传输，转换时处理
}

impl From<TensorValueWrapper> for TensorValue {
    fn from(wrapper: TensorValueWrapper) -> Self {
        match wrapper.dtype.as_str() {
            "float32" => {
                let array = ndarray::ArrayD::from_shape_vec(wrapper.shape, wrapper.data_f32).unwrap();
                TensorValue::Float32(array)
            },
            "int64" => {
                let array = ndarray::ArrayD::from_shape_vec(wrapper.shape, wrapper.data_i64).unwrap();
                TensorValue::Int64(array)
            },
            "bfloat16" => {
                // 将f32数据转换为bf16
                let bf16_data: Vec<half::bf16> = wrapper.data_bf16
                    .iter()
                    .map(|&x| half::bf16::from_f32(x))
                    .collect();
                let array = ndarray::ArrayD::from_shape_vec(wrapper.shape, bf16_data).unwrap();
                TensorValue::BFloat16(array)
            },
            "float16" => {
                // 将f32数据转换为f16
                let f16_data: Vec<half::f16> = wrapper.data_f16
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect();
                let array = ndarray::ArrayD::from_shape_vec(wrapper.shape, f16_data).unwrap();
                TensorValue::Float16(array)
            },
            _ => panic!("Unsupported dtype: {}", wrapper.dtype),
        }
    }
}

impl From<TensorValue> for TensorValueWrapper {
    fn from(value: TensorValue) -> Self {
        match value {
            TensorValue::Float32(array) => {
                let shape = array.shape().to_vec();
                let (data, _) = array.into_raw_vec_and_offset();
                TensorValueWrapper {
                    dtype: "float32".to_string(),
                    shape,
                    data_f32: data,
                    data_i64: Vec::new(),
                    data_bf16: Vec::new(),
                    data_f16: Vec::new(),
                }
            },
            TensorValue::Int64(array) => {
                let shape = array.shape().to_vec();
                let (data, _) = array.into_raw_vec_and_offset();
                TensorValueWrapper {
                    dtype: "int64".to_string(),
                    shape,
                    data_f32: Vec::new(),
                    data_i64: data,
                    data_bf16: Vec::new(),
                    data_f16: Vec::new(),
                }
            },
            TensorValue::BFloat16(array) => {
                let shape = array.shape().to_vec();
                let (data, _) = array.into_raw_vec_and_offset();
                // 将bf16数据转换为f32以便传输
                let f32_data: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
                TensorValueWrapper {
                    dtype: "bfloat16".to_string(),
                    shape,
                    data_f32: Vec::new(),
                    data_i64: Vec::new(),
                    data_bf16: f32_data,
                    data_f16: Vec::new(),
                }
            },
            TensorValue::Float16(array) => {
                let shape = array.shape().to_vec();
                let (data, _) = array.into_raw_vec_and_offset();
                // 将f16数据转换为f32以便传输
                let f32_data: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
                TensorValueWrapper {
                    dtype: "float16".to_string(),
                    shape,
                    data_f32: Vec::new(),
                    data_i64: Vec::new(),
                    data_bf16: Vec::new(),
                    data_f16: f32_data,
                }
            },
            _ => panic!("Unsupported TensorValue type"),
        }
    }
}

pub struct InferenceServer {
    model: Arc<ComputationGraph>,
    status: Arc<RwLock<ServerStatus>>,
}

impl InferenceServer {
    pub fn new(model_param_path: &str, model_weight_path: &str) -> Result<Self> {
        let model = ComputationGraph::from_pnnx(model_param_path, model_weight_path)?;
        Ok(Self {
            model: Arc::new(model),
            status: Arc::new(RwLock::new(ServerStatus { is_busy: false })),
        })
    }

    pub fn start(self, port: u16) -> Result<()> {
        let model: Arc<ComputationGraph> = self.model.clone();
        let status = self.status.clone();

        // 启动状态监控线程
        let status_thread = status.clone();
        thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let status_route = warp::path("status").and(warp::get()).map(move || {
                    let status = status_thread.read().unwrap();
                    warp::reply::json(&*status)
                });

                warp::serve(status_route).run(([127, 0, 0, 1], 3031)).await;
            });
        });

        // 启动主推理服务
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let infer_status = status.clone();
            let infer_model = model.clone();

            let infer_route = warp::path("infer")
                .and(warp::post())
                .and(warp::body::json())
                .and_then(move |request: InferenceRequest| {
                    let infer_status = infer_status.clone();
                    let infer_model = infer_model.clone();

                    async move {
                        // 检查服务器是否忙碌
                        {
                            let mut status = infer_status.write().unwrap();
                            if status.is_busy {
                                return Ok::<_, warp::Rejection>(warp::reply::json(
                                    &InferenceResponse {
                                        outputs: HashMap::new(),
                                        success: false,
                                        message: "Server is busy".to_string(),
                                        duration_ms: 0,
                                    },
                                ));
                            }
                            status.is_busy = true;
                        }

                        // 转换输入
                        let mut inputs = HashMap::new();
                        for (k, v) in request.inputs {
                            inputs.insert(k, TensorValue::from(v));
                        }

                        // 执行推理
                        let start = std::time::Instant::now(); // 新增开始计时
                        let result = match infer_model.compute(&inputs) {
                            Ok(outputs) => {
                                // 转换输出
                                let duration = start.elapsed().as_millis(); // 计算耗时
                                let mut response_outputs = HashMap::new();
                                for (k, v) in outputs {
                                    response_outputs.insert(k, TensorValueWrapper::from(v));
                                }
                                InferenceResponse {
                                    outputs: response_outputs,
                                    success: true,
                                    message: "Inference successful".to_string(),
                                    duration_ms: duration, // 填充耗时
                                }
                            }
                            Err(e) => {
                                let duration = start.elapsed().as_millis(); // 失败情况也记录时间
                                InferenceResponse {
                                    outputs: HashMap::new(),
                                    success: false,
                                    message: format!("Inference failed: {}", e),
                                    duration_ms: duration, // 填充耗时
                                }
                            }
                        };

                        // 更新服务器状态
                        {
                            let mut status = infer_status.write().unwrap();
                            status.is_busy = false;
                        }

                        Ok(warp::reply::json(&result))
                    }
                });

            println!("Inference server running at http://127.0.0.1:{}", port);
            warp::serve(infer_route).run(([127, 0, 0, 1], port)).await;
        });

        Ok(())
    }
}
