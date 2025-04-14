__kernel void conv2d(
    __global float* input,
    __global float* output,
    __global float* weights,
    __global float* bias,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int kernel_size_h,
    int kernel_size_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int output_height,
    int output_width
) {
    // 获取全局索引
    const int pos = get_global_id(0);

    // 计算输出的各个维度索引
    const int out_w = pos % output_width;
    const int out_h = (pos / output_width) % output_height;
    const int out_c = (pos / (output_width * output_height)) % output_channels;
    const int out_n = pos / (output_width * output_height * output_channels);

    // 检查是否越界
    if (out_n >= batch_size || out_c >= output_channels || out_h >= output_height || out_w >= output_width) {
        return;
    }

    // 计算组相关参数
    const int channels_per_group = input_channels / groups;
    const int filters_per_group = output_channels / groups;
    const int group_id = out_c / filters_per_group;
    const int in_c_start = group_id * channels_per_group;
    const int in_c_end = in_c_start + channels_per_group;

    // 计算输出索引
    const int output_idx = ((out_n * output_channels + out_c) * output_height + out_h) * output_width + out_w;

    float sum = 0.0f;
    // 对组内的每个输入通道进行循环累加
    for (int in_c = in_c_start; in_c < in_c_end; in_c++) {
        for (int kh = 0; kh < kernel_size_h; kh++) {
            for (int kw = 0; kw < kernel_size_w; kw++) {
                // 计算输入位置（考虑填充和膨胀）
                int in_h = out_h * stride_h + kh * dilation_h - padding_h;
                int in_w = out_w * stride_w + kw * dilation_w - padding_w;

                // 判断是否在输入边界内
                if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                    // 计算输入索引
                    int in_idx = ((out_n * input_channels + in_c) * input_height + in_h) * input_width + in_w;
                    // 计算权重索引（调整组内索引）
                    int w_idx = ((out_c * channels_per_group + (in_c - in_c_start)) * kernel_size_h + kh) * kernel_size_w + kw;
                    // 累加卷积结果
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
    }

    // 加上偏置
    sum += bias[out_c];

    // 将计算结果写入输出
    output[output_idx] = sum;
}