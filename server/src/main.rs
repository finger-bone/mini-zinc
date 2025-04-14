use anyhow::Result;
use mini_zinc::server::InferenceServer;

fn main() -> Result<()> {
    // 模型路径
    let resnet_param = "export/resnet/resnet18.pnnx.param";
    let resnet_weight = "export/resnet/resnet18.pnnx.bin";

    // 创建并启动推理服务器
    println!("正在初始化推理服务器...");
    let server = InferenceServer::new(resnet_param, resnet_weight)?;
    
    println!("推理服务器已启动");
    println!("- 推理接口: http://127.0.0.1:3030/infer (POST请求)");
    println!("- 状态接口: http://127.0.0.1:3031/status (GET请求)");
    println!("按Ctrl+C终止服务器");
    
    // 启动服务器（阻塞）
    server.start(3030)
}