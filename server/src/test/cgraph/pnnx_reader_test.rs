use crate::cgraph::{
    pnnx_reader::{PNNXKVType, PNNXReaderResult},
    pnnx_weight_reader::PNNXBinDataType,
};
use anyhow::Result;
use std::collections::HashMap;

#[test]
fn test_parse_maxpool2d() -> Result<()> {
    let line = "MaxPool2d MaxPool2d_0 1 1 1 1 stride=2 kernel_size=3 padding=1";
    let reader = PNNXReaderResult::from_text(&format!("7767517\n1 1\n{}", line))?;
    assert_eq!(reader.num_layers, 1);
    assert_eq!(reader.num_blobs, 1);
    assert_eq!(reader.lines.len(), 1);

    let pnnx_line = &reader.lines[0];
    assert_eq!(pnnx_line.op_type, "MaxPool2d");
    assert_eq!(pnnx_line.op_name, "MaxPool2d_0");
    assert_eq!(pnnx_line.input_idx_list, vec![1]);
    assert_eq!(pnnx_line.output_idx_list, vec![1]);
    Ok(())
}

#[test]
fn test_parse_relu() -> Result<()> {
    let line = "ReLU ReLU_0 1 1 1 1";
    let reader = PNNXReaderResult::from_text(&format!("7767517\n1 1\n{}", line))?;
    assert_eq!(reader.num_layers, 1);
    assert_eq!(reader.num_blobs, 1);
    assert_eq!(reader.lines.len(), 1);
    let pnnx_line = &reader.lines[0];
    assert_eq!(pnnx_line.op_type, "ReLU");
    assert_eq!(pnnx_line.op_name, "ReLU_0");
    assert_eq!(pnnx_line.input_idx_list, vec![1]);
    assert_eq!(pnnx_line.output_idx_list, vec![1]);
    Ok(())
}

#[test]
fn test_parse_conv2d() -> Result<()> {
    let line = "Conv2d Conv2d_0 1 1 1 1 in_channels=3 out_channels=64 kernel_size=3 stride=1 padding=1 dilation=1 groups=1 bias=1";
    let reader = PNNXReaderResult::from_text(&format!("7767517\n1 1\n{}", line))?;
    assert_eq!(reader.num_layers, 1);
    assert_eq!(reader.num_blobs, 1);
    assert_eq!(reader.lines.len(), 1);

    let pnnx_line = &reader.lines[0];
    assert_eq!(pnnx_line.op_type, "Conv2d");
    assert_eq!(pnnx_line.op_name, "Conv2d_0");
    assert_eq!(pnnx_line.input_idx_list, vec![1]);
    assert_eq!(pnnx_line.output_idx_list, vec![1]);
    Ok(())
}

#[test]
fn test_parse_expression() -> Result<()> {
    let line = "Expression Expression_0 2 1 1 2 1 expr=add(@0,@1)";
    let reader = PNNXReaderResult::from_text(&format!("7767517\n1 2\n{}", line))?;
    assert_eq!(reader.num_layers, 1);
    assert_eq!(reader.num_blobs, 2);
    assert_eq!(reader.lines.len(), 1);

    let pnnx_line = &reader.lines[0];
    assert_eq!(pnnx_line.op_type, "Expression");
    assert_eq!(pnnx_line.op_name, "Expression_0");
    assert_eq!(pnnx_line.input_idx_list, vec![1, 2]);
    assert_eq!(pnnx_line.output_idx_list, vec![1]);
    Ok(())
}

#[test]
fn test_get_shape_and_dtype_map() -> Result<()> {
    let text = r#"7767517
2 1
Conv2d Conv2d_0 1 1 1 1 @weight=(64,3,3,3)f32
ReLU ReLU_0 1 1 1 1 @output=(1,64,224,224)f32"#;
    let reader = PNNXReaderResult::from_text(text)?;

    let (shape_map, dtype_map) = reader.get_shape_and_dtype_map();
    assert_eq!(shape_map.len(), 2);

    let expected_shape_map: HashMap<String, Vec<usize>> = [
        ("Conv2d_0.weight".to_string(), vec![64, 3, 3, 3]),
        ("ReLU_0.output".to_string(), vec![1, 64, 224, 224]),
    ]
    .into_iter()
    .collect();

    assert_eq!(shape_map, expected_shape_map);

    let expected_dtype_map: HashMap<String, PNNXBinDataType> = [
        ("Conv2d_0.weight".to_string(), PNNXBinDataType::Float32),
        ("ReLU_0.output".to_string(), PNNXBinDataType::Float32),
    ]
    .into_iter()
    .collect();

    assert_eq!(dtype_map, expected_dtype_map);
    Ok(())
}

#[test]
fn test_parse_kvs() -> Result<()> {
    let line = "Conv2d Conv2d_0 1 1 1 1 in_channels=3 out_channels=64 kernel_size=3 stride=1 padding=1 dilation=1 groups=1 bias=1 @weight=1 #shape=64,3,3,3 $input=input0";
    let reader = PNNXReaderResult::from_text(&format!("7767517\n1 1\n{}", line))?;

    let pnnx_line = &reader.lines[0];
    assert_eq!(pnnx_line.kvs.len(), 11);

    // Test Attr type KVs
    let attr_kvs: Vec<_> = pnnx_line
        .kvs
        .iter()
        .filter(|kv| matches!(kv.kv_type, PNNXKVType::Attr))
        .collect();
    assert_eq!(attr_kvs.len(), 8);
    assert_eq!(attr_kvs[0].key, "in_channels");
    assert_eq!(attr_kvs[0].value, "3");

    // Test Tensor type KV
    let tensor_kv = pnnx_line
        .kvs
        .iter()
        .find(|kv| matches!(kv.kv_type, PNNXKVType::Tensor))
        .unwrap();
    assert_eq!(tensor_kv.key, "weight");
    assert_eq!(tensor_kv.value, "1");

    // Test Shape type KV
    let shape_kv = pnnx_line
        .kvs
        .iter()
        .find(|kv| matches!(kv.kv_type, PNNXKVType::Shape))
        .unwrap();
    assert_eq!(shape_kv.key, "shape");
    assert_eq!(shape_kv.value, "64,3,3,3");

    // Test Input type KV
    let input_kv = pnnx_line
        .kvs
        .iter()
        .find(|kv| matches!(kv.kv_type, PNNXKVType::Input))
        .unwrap();
    assert_eq!(input_kv.key, "input");
    assert_eq!(input_kv.value, "input0");

    Ok(())
}

#[test]
fn test_parse_resnet_block() -> Result<()> {
    let pnnx_text = r#"7767517
6 7
nn.Conv2d convbn2d_0 1 1 0 1 bias=True dilation=(1,1) groups=1 in_channels=3 kernel_size=(7,7) out_channels=64 padding=(3,3) padding_mode=zeros stride=(2,2) @bias=(64)f32 @weight=(64,3,7,7)f32 $input=0 #0=(1,3,224,224)f32 #1=(1,64,112,112)f32
nn.ReLU model.relu 1 1 1 2 #1=(1,64,112,112)f32 #2=(1,64,112,112)f32
nn.MaxPool2d model.maxpool 1 1 2 3 ceil_mode=False dilation=(1,1) kernel_size=(3,3) padding=(1,1) return_indices=False stride=(2,2) #2=(1,64,112,112)f32 #3=(1,64,56,56)f32
nn.Conv2d convbn2d_1 1 1 3 4 bias=True dilation=(1,1) groups=1 in_channels=64 kernel_size=(3,3) out_channels=64 padding=(1,1) padding_mode=zeros stride=(1,1) @bias=(64)f32 @weight=(64,64,3,3)f32 $input=3 #3=(1,64,56,56)f32 #4=(1,64,56,56)f32
nn.ReLU model.layer1.0.relu 1 1 4 5 #4=(1,64,56,56)f32 #5=(1,64,56,56)f32
pnnx.Expression pnnx_expr_14 2 1 6 3 7 expr=add(@0,@1) #6=(1,64,56,56)f32 #3=(1,64,56,56)f32 #7=(1,64,56,56)f32"#;

    let reader = PNNXReaderResult::from_text(pnnx_text)?;
    assert_eq!(reader.num_layers, 6);
    assert_eq!(reader.num_blobs, 7);
    assert_eq!(reader.lines.len(), 6);

    // Test Conv2d
    let conv2d = &reader.lines[0];
    assert_eq!(conv2d.op_type, "nn.Conv2d");
    assert_eq!(conv2d.op_name, "convbn2d_0");
    let conv2d_kvs: Vec<_> = conv2d.kvs.iter().collect();
    assert!(
        conv2d_kvs
            .iter()
            .any(|kv| kv.key == "kernel_size" && kv.value == "(7,7)")
    );
    assert!(
        conv2d_kvs
            .iter()
            .any(|kv| kv.key == "stride" && kv.value == "(2,2)")
    );
    assert!(
        conv2d_kvs
            .iter()
            .any(|kv| matches!(kv.kv_type, PNNXKVType::Shape))
    );

    // Test ReLU
    let relu = &reader.lines[1];
    assert_eq!(relu.op_type, "nn.ReLU");
    assert_eq!(relu.op_name, "model.relu");
    assert!(
        relu.kvs
            .iter()
            .any(|kv| matches!(kv.kv_type, PNNXKVType::Shape))
    );

    // Test MaxPool2d
    let maxpool = &reader.lines[2];
    assert_eq!(maxpool.op_type, "nn.MaxPool2d");
    assert_eq!(maxpool.op_name, "model.maxpool");
    let maxpool_kvs: Vec<_> = maxpool.kvs.iter().collect();
    assert!(
        maxpool_kvs
            .iter()
            .any(|kv| kv.key == "kernel_size" && kv.value == "(3,3)")
    );
    assert!(
        maxpool_kvs
            .iter()
            .any(|kv| kv.key == "stride" && kv.value == "(2,2)")
    );

    // Test Expression
    let expr = &reader.lines[5];
    assert_eq!(expr.op_type, "pnnx.Expression");
    assert_eq!(expr.op_name, "pnnx_expr_14");
    assert_eq!(expr.input_idx_list, vec![6, 3]);
    assert_eq!(expr.output_idx_list, vec![7]);
    assert!(
        expr.kvs
            .iter()
            .any(|kv| kv.key == "expr" && kv.value == "add(@0,@1)")
    );

    Ok(())
}
