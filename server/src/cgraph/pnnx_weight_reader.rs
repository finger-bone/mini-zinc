use byteorder::{LittleEndian, ReadBytesExt};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use zip::ZipArchive;

use crate::op::dtype::TensorValue;

use crate::op::dtype::DataType;

pub fn load_pnnx_zip_bin<P: AsRef<Path>>(
    zip_path: P,
    shape_map: &HashMap<String, Vec<usize>>,
    dtype_map: &HashMap<String, DataType>,
) -> std::io::Result<HashMap<String, TensorValue>> {
    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut tensor_map: HashMap<String, TensorValue> = HashMap::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let name = file.name().to_string();

        let shape = match shape_map.get(&name) {
            Some(s) => s.clone(),
            None => {
                eprintln!("Warning: no shape info for {}", name);
                continue;
            }
        };

        let dtype = dtype_map.get(&name).expect("Missing dtype");
        let numel: usize = shape.iter().product();
        let bytes_per_element = match dtype {
            DataType::Float32 => 4,
            DataType::Float16 | DataType::BFloat16 => 2,
            DataType::Boolean => 1,
            DataType::Int64 => 8,
        };

        let mut buf = vec![0u8; numel * bytes_per_element];
        file.read_exact(&mut buf)?;
        let mut rdr = Cursor::new(buf);

        let tensor_value = match dtype {
            DataType::Float32 => {
                let mut data = Vec::with_capacity(numel);
                for _ in 0..numel {
                    let val = rdr.read_f32::<LittleEndian>()?;
                    data.push(val);
                }
                TensorValue::Float32(ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap())
            }

            DataType::Float16 => {
                let mut data = Vec::<f16>::with_capacity(numel);
                for _ in 0..numel {
                    let bits = rdr.read_u16::<LittleEndian>()?;
                    let val = f16::from_bits(bits);
                    data.push(val);
                }
                TensorValue::Float16(ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap())
            }
            DataType::BFloat16 => {
                let mut data = Vec::<bf16>::with_capacity(numel);
                for _ in 0..numel {
                    let bits = rdr.read_u16::<LittleEndian>()?;
                    // convert into bf16
                    let val = bf16::from_bits(bits);
                    data.push(val);
                }
                TensorValue::BFloat16(ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap())
            }

            DataType::Boolean => {
                let mut data = Vec::with_capacity(numel);
                for _ in 0..numel {
                    let byte = rdr.read_u8()?;
                    data.push(byte != 0);
                }
                TensorValue::Boolean(ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap())
            }

            DataType::Int64 => {
                let mut data = Vec::with_capacity(numel);
                for _ in 0..numel {
                    let val = rdr.read_i64::<LittleEndian>()?;
                    data.push(val);
                }
                TensorValue::Int64(ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap())
            }
        };

        tensor_map.insert(name, tensor_value);
    }

    Ok(tensor_map)
}
