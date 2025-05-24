use anyhow::{Ok, Result};

use super::{conf::{TensorSplitConf, ToLayer}, dtype::TensorValue, layer::Forward};


pub struct TensorSplitLayer {
    pub lconf: TensorSplitConf
}

impl ToLayer for TensorSplitConf {
    fn to_layer(self) -> Result<Box<dyn Forward>> {
        Ok(Box::new(TensorSplitLayer {
            lconf: self
        }))
    }
}

impl Forward for TensorSplitLayer {
    fn forward(&mut self, inputs: &Vec<TensorValue>) -> Result<Vec<TensorValue>> {
        let input = &inputs[0];
        let dim = if self.lconf.dim < 0 {
            (input.shape().len() as isize + self.lconf.dim) as usize
        } else {
            self.lconf.dim as usize
        };
        let indices = &self.lconf.indices;
        let shape = input.shape();
        let total = shape[dim];
        let mut prev = 0;
        let mut splits = Vec::new();
        for &idx in indices.iter() {
            splits.push((prev, idx));
            prev = idx;
        }
        splits.push((prev, total));
        match input {
            TensorValue::Float32(arr) => {
                let mut result = Vec::new();
                for (start, end) in splits {
                    let mut slice_info = vec![ndarray::SliceInfoElem::Slice{ start: 0, end: None, step: 1 }; shape.len()];
                    slice_info[dim] = ndarray::SliceInfoElem::Slice{ start: start as isize, end: Some(end as isize), step: 1 };
                    let slice = unsafe { ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn>::new(slice_info).unwrap() };
                    let view = arr.slice(slice.as_ref());
                    result.push(TensorValue::Float32(view.to_owned()));
                }
                Ok(result)
            }
            TensorValue::Int64(arr) => {
                let mut result = Vec::new();
                for (start, end) in splits {
                    let mut slice_info = vec![ndarray::SliceInfoElem::Slice{ start: 0, end: None, step: 1 }; shape.len()];
                    slice_info[dim] = ndarray::SliceInfoElem::Slice{ start: start as isize, end: Some(end as isize), step: 1 };
                    let slice = unsafe { ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn>::new(slice_info).unwrap() };
                    let view = arr.slice(slice.as_ref());
                    result.push(TensorValue::Int64(view.to_owned()));
                }
                Ok(result)
            }
            TensorValue::Boolean(arr) => {
                let mut result = Vec::new();
                for (start, end) in splits {
                    let mut slice_info = vec![ndarray::SliceInfoElem::Slice{ start: 0, end: None, step: 1 }; shape.len()];
                    slice_info[dim] = ndarray::SliceInfoElem::Slice{ start: start as isize, end: Some(end as isize), step: 1 };
                    let slice = unsafe { ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn>::new(slice_info).unwrap() };
                    let view = arr.slice(slice.as_ref());
                    result.push(TensorValue::Boolean(view.to_owned()));
                }
                Ok(result)
            }
            TensorValue::BFloat16(arr) => {
                let mut result = Vec::new();
                for (start, end) in splits {
                    let mut slice_info = vec![ndarray::SliceInfoElem::Slice{ start: 0, end: None, step: 1 }; shape.len()];
                    slice_info[dim] = ndarray::SliceInfoElem::Slice{ start: start as isize, end: Some(end as isize), step: 1 };
                    let slice = unsafe { ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn>::new(slice_info).unwrap() };
                    let view = arr.slice(slice.as_ref());
                    result.push(TensorValue::BFloat16(view.to_owned()));
                }
                Ok(result)
            }
            TensorValue::Float16(arr) => {
                let mut result = Vec::new();
                for (start, end) in splits {
                    let mut slice_info = vec![ndarray::SliceInfoElem::Slice{ start: 0, end: None, step: 1 }; shape.len()];
                    slice_info[dim] = ndarray::SliceInfoElem::Slice{ start: start as isize, end: Some(end as isize), step: 1 };
                    let slice = unsafe { ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn>::new(slice_info).unwrap() };
                    let view = arr.slice(slice.as_ref());
                    result.push(TensorValue::Float16(view.to_owned()));
                }
                Ok(result)
            }
        }
    }
}