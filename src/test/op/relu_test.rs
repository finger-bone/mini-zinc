

use ndarray::ArrayD;

use crate::op::{
    conf::{FromZOpConf, ReLUConf, ZOpConf},
    layer::Forward,
};

#[test]
fn test_relu_forward() {
    let relu = ReLUConf { threshold: 0.0 };
    let layer = ZOpConf::ReLU(relu);
    let layer = ReLUConf::from_zopconf(layer).unwrap();

    // 测试正数保持不变
    let input = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = layer.forward(&vec![input.clone()]);
    assert_eq!(output[0], input);

    // 测试负数变为0
    let input = ArrayD::from_shape_vec(vec![2, 2], vec![-1.0, -2.0, 0.0, 1.0]).unwrap();
    let output = layer.forward(&vec![input]);
    let expected = ArrayD::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 0.0, 1.0]).unwrap();
    assert_eq!(output[0], expected);

    // 测试自定义阈值
    let relu = ReLUConf { threshold: 2.0 };
    let layer = ZOpConf::ReLU(relu);
    let layer = ReLUConf::from_zopconf(layer).unwrap();
    let input = ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = layer.forward(&vec![input]);
    let expected = ArrayD::from_shape_vec(vec![2, 2], vec![0.0, 0.0, 3.0, 4.0]).unwrap();
    assert_eq!(output[0], expected);
}

#[test]
fn test_relu_config() {
    // 测试正确的配置
    let relu = ReLUConf { threshold: 0.0 };
    let layer = ZOpConf::ReLU(relu);
    assert!(ReLUConf::from_zopconf(layer).is_ok());

    // 测试错误的配置类型
    let layer = ZOpConf::ReLU(ReLUConf { threshold: 0.0 });
    assert!(ReLUConf::from_zopconf(layer).is_ok());
}
