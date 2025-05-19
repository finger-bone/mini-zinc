#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__kernel 
void linear(__global float *input, __global float *output, __global float *weights, __global float *bias,
          const int batch_size, const int in_features, const int out_features) 
{
    const int pos = get_global_id(0);
    const int out_idx = pos % out_features;
    const int batch_idx = pos / out_features;
    
    // Bounds check
    if (batch_idx >= batch_size || out_idx >= out_features) {
        return;
    }
    
    // Initialize accumulator with bias
    float sum = bias[out_idx];
    
    // Process input in tiles
    for (int tile = 0; tile < in_features; tile += TILE_SIZE) {
        // Calculate actual tile size (handle edge cases)
        const int current_tile_size = min(TILE_SIZE, in_features - tile);
        
        // Load input tile to local memory
        float input_tile[TILE_SIZE];
        for (int i = 0; i < current_tile_size; i++) {
            input_tile[i] = input[batch_idx * in_features + tile + i];
        }
        
        // Load weights tile to local memory
        float weight_tile[TILE_SIZE];
        for (int i = 0; i < current_tile_size; i++) {
            weight_tile[i] = weights[out_idx * in_features + tile + i];
        }
        
        // Compute dot product for current tile
        for (int i = 0; i < current_tile_size; i++) {
            sum += input_tile[i] * weight_tile[i];
        }
    }
    
    // Write final result
    output[pos] = sum;
}