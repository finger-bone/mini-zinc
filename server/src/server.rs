use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use warp::Filter;
use tokio::sync::{mpsc, oneshot, RwLock}; // Use tokio's RwLock for async context

use crate::cgraph::graph::ComputationGraph;
use crate::op::dtype::TensorValue;


// 服务器状态结构体
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerStatus {
    pub is_busy: bool,
    // Add other relevant server status fields here if needed
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
    pub duration_ms: u128,
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
                let array =
                    ndarray::ArrayD::from_shape_vec(wrapper.shape, wrapper.data_f32).unwrap();
                TensorValue::Float32(array)
            }
            "int64" => {
                let array =
                    ndarray::ArrayD::from_shape_vec(wrapper.shape, wrapper.data_i64).unwrap();
                TensorValue::Int64(array)
            }
            "bfloat16" => {
                // 将f32数据转换为bf16
                let bf16_data: Vec<half::bf16> = wrapper
                    .data_bf16
                    .iter()
                    .map(|&x| half::bf16::from_f32(x))
                    .collect();
                let array = ndarray::ArrayD::from_shape_vec(wrapper.shape, bf16_data).unwrap();
                TensorValue::BFloat16(array)
            }
            "float16" => {
                // 将f32数据转换为f16
                let f16_data: Vec<half::f16> = wrapper
                    .data_f16
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect();
                let array = ndarray::ArrayD::from_shape_vec(wrapper.shape, f16_data).unwrap();
                TensorValue::Float16(array)
            }
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
            }
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
            }
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
            }
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
            }
            _ => panic!("Unsupported TensorValue type"),
        }
    }
}

// Message type for sending inference requests to the dedicated thread
struct InferenceRequestMessage {
    request: InferenceRequest,
    response_sender: oneshot::Sender<InferenceResponse>,
}


pub struct InferenceServer {
    // Sender to send requests to the dedicated inference thread
    request_sender: mpsc::Sender<InferenceRequestMessage>,
    // Server status, managed by the main server loop
    status: Arc<RwLock<ServerStatus>>,
}

impl InferenceServer {
    pub fn new(model_param_path: &str, model_weight_path: &str) -> Result<Self> {
        let (request_sender, request_receiver) = mpsc::channel::<InferenceRequestMessage>(100);

        let model_param_path_clone = model_param_path.to_string();
        let model_weight_path_clone = model_weight_path.to_string();

        std::thread::spawn(move || {
            let model = match ComputationGraph::from_pnnx(
                &model_param_path_clone,
                &model_weight_path_clone,
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Failed to load ComputationGraph in inference thread: {}", e);
                    return;
                }
            };

            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create Tokio runtime for inference worker");

            rt.block_on(async {
                Self::inference_worker(model, request_receiver).await;
            });
        });

        Ok(Self {
            request_sender,
            status: Arc::new(RwLock::new(ServerStatus { is_busy: false })),
        })
    }

    async fn inference_worker(
        mut model: ComputationGraph,
        mut receiver: mpsc::Receiver<InferenceRequestMessage>,
    ) {
        println!("Inference worker thread started.");
        while let Some(msg) = receiver.recv().await {
            let InferenceRequestMessage {
                request,
                response_sender,
            } = msg;

            let start = Instant::now();
            let mut inputs = HashMap::new();
            for (k, v) in request.inputs {
                inputs.insert(k, TensorValue::from(v));
            }

            let result = match model.compute(&inputs) {
                Ok(outputs) => {
                    let duration = start.elapsed().as_millis();
                    let mut response_outputs = HashMap::new();
                    for (k, v) in outputs {
                        response_outputs.insert(k, TensorValueWrapper::from(v));
                    }
                    InferenceResponse {
                        outputs: response_outputs,
                        success: true,
                        message: "Inference successful".to_string(),
                        duration_ms: duration,
                    }
                }
                Err(e) => {
                    let duration = start.elapsed().as_millis();
                    InferenceResponse {
                        outputs: HashMap::new(),
                        success: false,
                        message: format!("Inference failed: {}", e),
                        duration_ms: duration,
                    }
                }
            };

            if let Err(_) = response_sender.send(result) {
                eprintln!("Failed to send inference response back to request handler.");
            }
        }
        println!("Inference worker thread shutting down.");
    }

    pub fn start(self, port: u16) -> Result<()> {
        let request_sender = self.request_sender.clone();
        let server_status = self.status.clone();

        // Status route - CHANGED TO and_then
        let status_route = warp::path("status")
            .and(warp::get())
            .and_then(move || { // Use and_then for async operations
                let server_status_clone = server_status.clone();
                async move {
                    let status = server_status_clone.read().await;
                    // Wrap the reply in Ok for and_then
                    Ok::<_, warp::Rejection>(warp::reply::json(&*status))
                }
            });

        // Inference route (already using and_then correctly)
        let infer_sender_clone = request_sender.clone();
        let infer_status_clone = self.status.clone();

        let infer_route = warp::path("infer")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(move |request: InferenceRequest| {
                let infer_sender = infer_sender_clone.clone();
                let infer_status = infer_status_clone.clone();

                async move {
                    let (response_tx, response_rx) = oneshot::channel::<InferenceResponse>();

                    {
                        let mut status_guard = infer_status.write().await;
                        if status_guard.is_busy {
                            return Ok::<_, warp::Rejection>(warp::reply::json(
                                &InferenceResponse {
                                    outputs: HashMap::new(),
                                    success: false,
                                    message: "Server is busy".to_string(),
                                    duration_ms: 0,
                                },
                            ));
                        }
                        status_guard.is_busy = true;
                    }

                    let send_result = infer_sender
                        .send(InferenceRequestMessage {
                            request,
                            response_sender: response_tx,
                        })
                        .await;

                    let inference_response = match send_result {
                        Ok(_) => {
                            response_rx.await.unwrap_or_else(|_| {
                                InferenceResponse {
                                    outputs: HashMap::new(),
                                    success: false,
                                    message: "Inference worker failed to send response".to_string(),
                                    duration_ms: 0,
                                }
                            })
                        }
                        Err(e) => {
                            eprintln!("Failed to send inference request to worker: {}", e);
                            InferenceResponse {
                                outputs: HashMap::new(),
                                success: false,
                                message: "Server internal error: inference worker unavailable".to_string(),
                                duration_ms: 0,
                            }
                        }
                    };

                    {
                        let mut status_guard = infer_status.write().await;
                        status_guard.is_busy = false;
                    }

                    Ok(warp::reply::json(&inference_response))
                }
            });

        let routes = infer_route.or(status_route);

        println!("Inference server running at http://127.0.0.1:{}", port);

        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?
            .block_on(async {
                warp::serve(routes).run(([127, 0, 0, 1], port)).await;
            });

        Ok(())
    }
}