__kernel 
void linear(__global float *input, __global float *output, __global float *weights, __global float *bias,
          const int batch_size, const int in_features, const int out_features) 
{
    // Get global position
    const int pos = get_global_id(0);
    
    // Calculate output indices
    const int out_idx = pos % out_features;
    const int batch_idx = pos / out_features;
    
    // Check bounds
    if (batch_idx >= batch_size || out_idx >= out_features) {
        return;
    }
    
    // Initialize with bias
    float sum = bias[out_idx];
    
    // Matrix multiplication (dot product)
    for (int i = 0; i < in_features; i++) {
        sum += input[batch_idx * in_features + i] * weights[out_idx * in_features + i];
    }
    
    // Write output
    output[pos] = sum;
}