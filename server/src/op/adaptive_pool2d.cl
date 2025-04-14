__kernel 
void adaptive_pool(__global float *input, __global float *output,
         const int batch_size, const int channels, const int input_height, const int input_width,
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

    // Compute input region for this output cell (adaptive logic)
    const int in_h_start = (int)floor((float)(out_h * input_height) / output_height);
    const int in_h_end   = (int)ceil((float)((out_h + 1) * input_height) / output_height);
    const int in_w_start = (int)floor((float)(out_w * input_width) / output_width);
    const int in_w_end   = (int)ceil((float)((out_w + 1) * input_width) / output_width);

    // Initialize accumulator
    float acc = (pool_type == 0) ? -FLT_MAX : 0.0f;
    int count = 0;

    // Perform pooling over the region
    for (int in_h = in_h_start; in_h < in_h_end; ++in_h) {
        for (int in_w = in_w_start; in_w < in_w_end; ++in_w) {
            int in_idx = ((out_n * channels + out_c) * input_height + in_h) * input_width + in_w;
            float val = input[in_idx];

            if (pool_type == 0) {
                acc = fmax(acc, val);
            } else {
                acc += val;
                count++;
            }
        }
    }

    // Finalize average pooling
    if (pool_type == 1 && count > 0) {
        acc /= count;
    }

    // Write result
    output[pos] = acc;
}