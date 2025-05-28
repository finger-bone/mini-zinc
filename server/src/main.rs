use anyhow::Result;
use mini_zinc::server::InferenceServer;

fn start_resnet() -> Result<()> {
    // 模型路径
    let resnet_param = "export/resnet/resnet18.pnnx.param";
    let resnet_weight = "export/resnet/resnet18.pnnx.bin";
    let server = InferenceServer::new(resnet_param, resnet_weight)?;
    // 启动服务器（阻塞）
    server.start(3030)
}

fn start_bert() -> Result<()> {
    // 模型路径
    let bert_param = "export/bert/distilbert-base-uncased-base-fillmask.pnnx.param";
    let bert_weight = "export/bert/distilbert-base-uncased-base-fillmask.pnnx.bin";
    let server = InferenceServer::new(bert_param, bert_weight)?;
    // 启动服务器（阻塞）
    server.start(3030)
}

fn start_smollm() -> Result<()> {
    // 模型路径
    let smollm_param = "export/smollm/smollm.pnnx.param";
    let smollm_weight = "export/smollm/smollm.pnnx.bin";
    let server = InferenceServer::new(smollm_param, smollm_weight)?;
    // 启动服务器（阻塞）
    server.start(3030)
}

fn start_yolo() -> Result<()> {
    // 模型路径
    let yolo_param = "export/yolo/yolov5nu.pnnx.param";
    let yolo_weight = "export/yolo/yolov5nu.pnnx.bin";
    let server = InferenceServer::new(yolo_param, yolo_weight)?;
    // 启动服务器（阻塞）
    server.start(3030)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    println!("- 推理接口: http://127.0.0.1:3030/infer (POST请求)");
    println!("- 状态接口: http://127.0.0.1:3031/status (GET请求)");
    if args.len() > 1 && args[1] == "bert" {
        println!("- 启动BERT模型");
        start_bert()
    } else if args.len() > 1 && args[1] == "smollm" {
        println!("- 启动SMOLLM模型");
        start_smollm()
    } else if args.len() > 1 && args[1] == "yolo" {
        println!("- 启动YOLO模型");
        start_yolo()
    } else if args.len() > 1 && args[1] == "resnet" {
        println!("- 启动ResNet模型");
        start_resnet()
    } else {
        println!("请输入模型名称: resnet, bert, smollm, yolo");
        Ok(())
    }
}
