use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Cursor};
use std::path::Path;
use zip::ZipArchive;
use ndarray::{ArrayD, IxDyn};
use byteorder::{LittleEndian, ReadBytesExt};

pub fn load_pnnx_zip_bin<P: AsRef<Path>>(
    zip_path: P,
    shape_map: &HashMap<String, Vec<usize>>,
) -> std::io::Result<HashMap<String, ArrayD<f32>>> {
    let file = File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut tensor_map = HashMap::new();

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

        // 读取所有数据为 f32
        let numel: usize = shape.iter().product();
        let mut buf = vec![0u8; numel * 4];
        file.read_exact(&mut buf)?;

        // 解析为 f32
        let mut floats = Vec::with_capacity(numel);
        let mut rdr = Cursor::new(buf);
        for _ in 0..numel {
            let val = rdr.read_f32::<LittleEndian>()?;
            floats.push(val);
        }

        // 构建 ndarray
        let array = ArrayD::from_shape_vec(IxDyn(&shape), floats)
            .expect("shape and data mismatch");

        tensor_map.insert(name, array);
    }

    Ok(tensor_map)
}