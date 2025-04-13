use ndarray::ArrayD;

pub trait Forward {
    fn forward(&self, input: &Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>>;
}
