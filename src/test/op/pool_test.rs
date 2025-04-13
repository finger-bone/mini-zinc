
use ndarray::ArrayD;

use crate::op::{conf::{FromZOpConf, PoolConf, PoolType, ZOpConf}, layer::Forward};

#[test]
fn test_maxpool_forward() {
    // Create a simple 1x1x4x4 input (batch_size=1, channels=1, height=4, width=4)
    let input = ArrayD::from_shape_vec(
        vec![1, 1, 4, 4], 
        vec![1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0, 
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0]
    ).unwrap();
    
    // Create a 2x2 max pooling layer
    let pool = PoolConf {
        kernel_size: vec![2, 2],
        stride: vec![2, 2],
        padding: vec![0, 0],
        pool_type: PoolType::Max,
    };
    
    let layer = ZOpConf::Pool(pool);
    let layer = PoolConf::from_zopconf(layer).unwrap();
    
    // Forward pass
    let output = layer.forward(&vec![input]);
    
    // Check output shape (should be 1x1x2x2)
    assert_eq!(output[0].shape(), &[1, 1, 2, 2]);
    
    // Check max pooling values
    let expected = ArrayD::from_shape_vec(
        vec![1, 1, 2, 2], 
        vec![6.0, 8.0, 14.0, 16.0]
    ).unwrap();
    
    // Due to potential floating-point precision issues, we'll check approximately
    for i in 0..output[0].len() {
        assert!((output[0].as_slice().unwrap()[i] - expected.as_slice().unwrap()[i]).abs() < 1e-5);
    }
}

#[test]
fn test_avgpool_forward() {
    // Create a simple 1x1x4x4 input (batch_size=1, channels=1, height=4, width=4)
    let input = ArrayD::from_shape_vec(
        vec![1, 1, 4, 4], 
        vec![1.0, 2.0, 3.0, 4.0, 
                5.0, 6.0, 7.0, 8.0, 
                9.0, 10.0, 11.0, 12.0, 
                13.0, 14.0, 15.0, 16.0]
    ).unwrap();
    
    // Create a 2x2 average pooling layer
    let pool = PoolConf {
        kernel_size: vec![2, 2],
        stride: vec![2, 2],
        padding: vec![0, 0],
        pool_type: PoolType::Avg,
    };
    
    let layer = ZOpConf::Pool(pool);
    let layer = PoolConf::from_zopconf(layer).unwrap();
    
    // Forward pass
    let output = layer.forward(&vec![input]);
    
    // Check output shape (should be 1x1x2x2)
    assert_eq!(output[0].shape(), &[1, 1, 2, 2]);
    
    // Check average pooling values
    let expected = ArrayD::from_shape_vec(
        vec![1, 1, 2, 2], 
        vec![3.5, 5.5, 11.5, 13.5]
    ).unwrap();
    
    // Due to potential floating-point precision issues, we'll check approximately
    for i in 0..output[0].len() {
        assert!((output[0].as_slice().unwrap()[i] - expected.as_slice().unwrap()[i]).abs() < 1e-5);
    }
}
