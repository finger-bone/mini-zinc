__kernel 
void conv2d(__global float *input, __global float *output, __global float *weights, __global float *bias,
           const int batch_size, const int input_channels, const int input_height, const int input_width,
           const int filters, const int kernel_height, const int kernel_width,
           const int stride_h, const int stride_w, const int pad_h, const int pad_w,
           const int dilation_h, const int dilation_w, const int groups,
           const int output_height, const int output_width) 
{
    // Get global position
    const int pos = get_global_id(0);
    
    // Calculate output indices
    const int out_w = pos % output_width;
    const int out_h = (pos / output_width) % output_height;
    const int out_c = (pos / (output_width * output_height)) % filters;
    const int out_n = pos / (output_width * output_height * filters);
    
    // Check bounds
    if (out_n >= batch_size || out_c >= filters || out_h >= output_height || out_w >= output_width) {
        return;
    }
    
    // Initialize accumulator with bias
    float acc = bias[out_c];
    
    // Calculate group parameters
    const int channels_per_group = input_channels / groups;
    const int filters_per_group = filters / groups;
    const int group_id = out_c / filters_per_group;
    const int in_c_start = group_id * channels_per_group;
    const int in_c_end = in_c_start + channels_per_group;
    
    // Convolve
    for (int in_c = in_c_start; in_c < in_c_end; in_c++) {
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                // Calculate input position with padding and dilation
                int in_h = out_h * stride_h + kh * dilation_h - pad_h;
                int in_w = out_w * stride_w + kw * dilation_w - pad_w;
                
                // Skip if outside input bounds (padding area)
                if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                    // Input index
                    int in_idx = ((out_n * input_channels + in_c) * input_height + in_h) * input_width + in_w;
                    
                    // Weight index - adjust for groups
                    int w_idx = ((out_c * channels_per_group + (in_c - in_c_start)) * kernel_height + kh) * kernel_width + kw;
                    
                    // Accumulate
                    acc += input[in_idx] * weights[w_idx];
                }
            }
        }
    }
    
    // Write output
    output[pos] = acc;
}