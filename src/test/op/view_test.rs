
use ndarray::ArrayD;
use crate::op::{
    conf::{FromZOpConf, ViewConf, ZOpConf},
    layer::Forward,
};

#[test]
fn test_view_forward() {
    // 测试2D到1D的重塑
    let view = ViewConf {
        input_shape: vec![2, 3],
        output_shape: vec![6],
    };
    let layer = ZOpConf::View(view);
    let layer = ViewConf::from_zopconf(layer).unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let output = layer.forward(&vec![input]);
    let expected = ArrayD::from_shape_vec(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(output[0], expected);

    // 测试3D到2D的重塑
    let view = ViewConf {
        input_shape: vec![2, 2, 2],
        output_shape: vec![4, 2],
    };
    let layer = ZOpConf::View(view);
    let layer = ViewConf::from_zopconf(layer).unwrap();

    let input = ArrayD::from_shape_vec(
        vec![2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ).unwrap();
    let output = layer.forward(&vec![input]);
    let expected = ArrayD::from_shape_vec(
        vec![4, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ).unwrap();
    assert_eq!(output[0], expected);
}

#[test]
#[should_panic(expected = "Input and output shapes must have the same number of elements")]
fn test_view_shape_mismatch() {
    let view = ViewConf {
        input_shape: vec![2, 3],
        output_shape: vec![5], // 元素数量不匹配
    };
    let layer = ZOpConf::View(view);
    let layer = ViewConf::from_zopconf(layer).unwrap();

    let input = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    layer.forward(&vec![input]); // 这里应该会panic
}

#[test]
fn test_view_config() {
    // 测试正确的配置
    let view = ViewConf {
        input_shape: vec![2, 3],
        output_shape: vec![6],
    };
    let layer = ZOpConf::View(view);
    assert!(ViewConf::from_zopconf(layer).is_ok());

    // 测试错误的配置类型
    let view = ViewConf {
        input_shape: vec![2, 3],
        output_shape: vec![6],
    };
    let layer = ZOpConf::View(view);
    assert!(ViewConf::from_zopconf(layer).is_ok());
}