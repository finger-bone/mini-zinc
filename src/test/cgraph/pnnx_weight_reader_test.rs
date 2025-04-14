use crate::cgraph::pnnx_weight_reader::{PNNXBinDataType, load_pnnx_zip_bin};
use crate::op::layer::TensorValue;
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Write};
use tempfile::tempdir;
use zip::write::FileOptions;
use zip::{CompressionMethod, ZipWriter};

#[test]
fn test_load_pnnx_zip_bin_multiple_weights() {
    // 创建一个临时目录
    let dir = tempdir().unwrap();
    let zip_path = dir.path().join("test.zip");
    let mut zip = ZipWriter::new(File::create(&zip_path).unwrap());

    // 准备第一个权重数据
    let test_data1 = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut buf1 = Vec::new();
    let mut cursor1 = Cursor::new(&mut buf1);
    for &val in &test_data1 {
        cursor1.write_f32::<LittleEndian>(val).unwrap();
    }

    // 准备第二个权重数据
    let test_data2 = vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
    let mut buf2 = Vec::new();
    let mut cursor2 = Cursor::new(&mut buf2);
    for &val in &test_data2 {
        cursor2.write_f32::<LittleEndian>(val).unwrap();
    }

    // 写入测试数据到zip文件
    let options: FileOptions<()> =
        FileOptions::default().compression_method(CompressionMethod::Stored);

    zip.start_file("weight1", options).unwrap();
    zip.write_all(&buf1).unwrap();
    zip.start_file("weight2", options).unwrap();
    zip.write_all(&buf2).unwrap();
    zip.finish().unwrap();

    // 构造shape_map
    let mut shape_map = HashMap::new();
    shape_map.insert("weight1".to_string(), vec![2, 2]);
    shape_map.insert("weight2".to_string(), vec![2, 3]);
    let mut data_type_map = HashMap::new();
    data_type_map.insert("weight1".to_string(), PNNXBinDataType::Float32);
    data_type_map.insert("weight2".to_string(), PNNXBinDataType::Float32);

    // 加载并验证结果
    let result = load_pnnx_zip_bin(zip_path.to_str().unwrap(), &shape_map, &data_type_map).unwrap();

    // 验证第一个权重
    assert!(result.contains_key("weight1"));
    let array1 = &result["weight1"];
    if let TensorValue::Float32(array1) = array1 {
        assert_eq!(array1.shape(), &[2, 2]);
        assert_eq!(array1.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    } else {
        panic!("Unexpected type for weight1");
    }

    // 验证第二个权重
    assert!(result.contains_key("weight2"));
    let array2 = &result["weight2"];
    if let TensorValue::Float32(array2) = array2 {
        assert_eq!(array2.shape(), &[2, 3]);
        assert_eq!(array2.as_slice().unwrap(), &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    } else {
        panic!("Unexpected type for weight2");
    }
}

#[test]
fn test_load_pnnx_zip_bin_compressed() {
    // 创建一个临时目录
    let dir = tempdir().unwrap();
    let zip_path = dir.path().join("test.zip");
    let mut zip = ZipWriter::new(File::create(&zip_path).unwrap());

    // 准备测试数据
    let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    for &val in &test_data {
        cursor.write_f32::<LittleEndian>(val).unwrap();
    }

    // 使用压缩方式写入测试数据到zip文件
    let options: FileOptions<()> =
        FileOptions::default().compression_method(CompressionMethod::Deflated);

    zip.start_file("weight1", options).unwrap();
    zip.write_all(&buf).unwrap();
    zip.finish().unwrap();

    // 构造shape_map
    let mut shape_map = HashMap::new();
    shape_map.insert("weight1".to_string(), vec![2, 2]);
    let mut data_type_map = HashMap::new();
    data_type_map.insert("weight1".to_string(), PNNXBinDataType::Float32);

    // 加载并验证结果
    let result = load_pnnx_zip_bin(zip_path.to_str().unwrap(), &shape_map, &data_type_map).unwrap();

    assert!(result.contains_key("weight1"));
    let array = &result["weight1"];
    if let TensorValue::Float32(array) = array {
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    } else {
        panic!("Unexpected type for weight1");
    }
}

#[test]
fn test_load_pnnx_zip_bin_shape_mismatch() {
    // 创建一个临时目录
    let dir = tempdir().unwrap();
    let zip_path = dir.path().join("test.zip");
    let mut zip = ZipWriter::new(File::create(&zip_path).unwrap());

    // 准备测试数据
    let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    for &val in &test_data {
        cursor.write_f32::<LittleEndian>(val).unwrap();
    }

    // 写入测试数据到zip文件
    let options: FileOptions<()> =
        FileOptions::default().compression_method(CompressionMethod::Stored);

    zip.start_file("weight1", options).unwrap();
    zip.write_all(&buf).unwrap();
    zip.finish().unwrap();

    // 构造错误的shape_map（元素数量不匹配）
    let mut shape_map = HashMap::new();
    shape_map.insert("weight1".to_string(), vec![3, 3]);
    let mut data_type_map = HashMap::new();
    data_type_map.insert("weight1".to_string(), PNNXBinDataType::Float32);

    // 加载并验证结果应该失败
    let result = load_pnnx_zip_bin(zip_path.to_str().unwrap(), &shape_map, &data_type_map);
    assert!(result.is_err());
}

#[test]
fn test_load_pnnx_zip_bin() {
    // 创建一个临时目录
    let dir = tempdir().unwrap();
    let zip_path = dir.path().join("test.zip");
    let mut zip = ZipWriter::new(File::create(&zip_path).unwrap());

    // 准备测试数据
    let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    for &val in &test_data {
        cursor.write_f32::<LittleEndian>(val).unwrap();
    }

    // 写入测试数据到zip文件
    let options: FileOptions<()> =
        FileOptions::default().compression_method(CompressionMethod::Stored);

    zip.start_file("weight1", options).unwrap();
    zip.write_all(&buf).unwrap();
    zip.finish().unwrap();

    // 构造shape_map
    let mut shape_map = HashMap::new();
    shape_map.insert("weight1".to_string(), vec![2, 2]);
    let mut data_type_map = HashMap::new();
    data_type_map.insert("weight1".to_string(), PNNXBinDataType::Float32);

    // 加载并验证结果
    let result = load_pnnx_zip_bin(zip_path.to_str().unwrap(), &shape_map, &data_type_map).unwrap();

    assert!(result.contains_key("weight1"));
    let array = &result["weight1"];
    if let TensorValue::Float32(array) = array {
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    } else {
        panic!("Unexpected type for weight1");
    }
}
