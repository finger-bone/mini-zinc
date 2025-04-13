use byteorder::{LittleEndian, ReadBytesExt};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use zip::ZipArchive;

use crate::op::layer::TensorValue;

#[derive(Debug, PartialEq)]
pub enum PNNXBinDataType {
    BFloat16,
    Float16,
    Float32,
}

pub fn load_pnnx_zip_bin<P: AsRef<Path>>(
    zip_path: P,
    shape_map: &HashMap<String, Vec<usize>>,
    dtype_map: &HashMap<String, PNNXBinDataType>,
) -> std::io::Result<HashMap<String, TensorValue>> {
    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut tensor_map: HashMap<String, TensorValue> = HashMap::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();

        // 获取 shape 信息（从 param 文件中解析来的）
        let shape = match shape_map.get(&name) {
            Some(s) => s.clone(),
            None => {
                eprintln!("Warning: no shape info for {}", name);
                continue;
            }
        };

        let bytes_per_element = match dtype_map.get(&name) {
            Some(PNNXBinDataType::BFloat16) | Some(PNNXBinDataType::Float16) => 2,
            Some(PNNXBinDataType::Float32) => 4,
            _ => 4,
        };
        let numel: usize = shape.iter().product();
        let mut buf = vec![0u8; numel * bytes_per_element];
        file.read_exact(&mut buf)?;

        let mut floats = Vec::with_capacity(numel);
        let mut rdr = Cursor::new(buf);
        match dtype_map.get(&name) {
            Some(PNNXBinDataType::Float32) => {
                for _ in 0..numel {
                    let val = rdr.read_f32::<LittleEndian>()?;
                    floats.push(val);
                }
            }
            Some(PNNXBinDataType::Float16) => {
                unimplemented!()
            }
            Some(PNNXBinDataType::BFloat16) => {
                unimplemented!()
            }
            _ => {
                unimplemented!()
            }
        }

        let array = ArrayD::from_shape_vec(IxDyn(&shape), floats).expect("shape and data mismatch");

        tensor_map.insert(name, TensorValue::Float32(array));
    }

    Ok(tensor_map)
}
