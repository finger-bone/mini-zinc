__kernel 
void pool(__global float *input, __global float *output,
         const int batch_size, const int channels, const int input_height, const int input_width,
         const int kernel_height, const int kernel_width,
         const int stride_h, const int stride_w, const int pad_h, const int pad_w,
         const int output_height, const int output_width, const int pool_type) 
{
    // Get global position
    const int pos = get_global_id(0);
    
    // Calculate output indices
    const int out_w = pos % output_width;
    const int out_h = (pos / output_width) % output_height;
    const int out_c = (pos / (output_width * output_height)) % channels;
    const int out_n = pos / (output_width * output_height * channels);
    
    // Check bounds
    if (out_n >= batch_size || out_c >= channels || out_h >= output_height || out_w >= output_width) {
        return;
    }
    
    // Initialize accumulator based on pool type
    float acc = (pool_type == 0) ? -FLT_MAX : 0.0f;
    int count = 0;
    
    // Perform pooling
    for (int kh = 0; kh < kernel_height; kh++) {
        for (int kw = 0; kw < kernel_width; kw++) {
            // Calculate input position with padding
            int in_h = out_h * stride_h + kh - pad_h;
            int in_w = out_w * stride_w + kw - pad_w;
            
            // Skip if outside input bounds (padding area)
            if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                // Input index
                int in_idx = ((out_n * channels + out_c) * input_height + in_h) * input_width + in_w;
                
                if (pool_type == 0) { // Max pooling
                    acc = max(acc, input[in_idx]);
                } else { // Average pooling
                    acc += input[in_idx];
                    count++;
                }
            }
        }
    }
    
    // Finalize average pooling
    if (pool_type == 1 && count > 0) {
        acc /= count;
    }
    
    // Write output
    output[pos] = acc;
}