__kernel 
void pool(__global float *input, __global float *output,
          const int batch_size, const int channels,
          const int input_height, const int input_width,
          const int kernel_height, const int kernel_width,
          const int stride_h, const int stride_w,
          const int pad_h, const int pad_w,
          const int output_height, const int output_width,
          const int pool_type) 
{
    const int pos = get_global_id(0);
    
    const int out_w = pos % output_width;
    const int out_h = (pos / output_width) % output_height;
    const int out_c = (pos / (output_width * output_height)) % channels;
    const int out_n = pos / (output_width * output_height * channels);

    if (out_n >= batch_size || out_c >= channels || out_h >= output_height || out_w >= output_width) {
        return;
    }

    float acc = (pool_type == 0) ? -FLT_MAX : 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernel_height; ++kh) {
        for (int kw = 0; kw < kernel_width; ++kw) {
            int in_h = out_h * stride_h - pad_h + kh;
            int in_w = out_w * stride_w - pad_w + kw;

            if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                int in_idx = ((out_n * channels + out_c) * input_height + in_h) * input_width + in_w;

                float val = input[in_idx];
                if (pool_type == 0) {
                    acc = fmax(acc, val);
                } else {
                    acc += val;
                    count += 1;
                }
            }
        }
    }

    if (pool_type == 1 && count > 0) {
        acc /= (float)count;
    }

    output[pos] = acc;
}