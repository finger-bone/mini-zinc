#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

inline float silu_func(float x) {
    return x / (1.0f + exp(-x));
}

__kernel 
void silu(__global float *input, __global float *output) 
{
    __local float input_tile[TILE_SIZE];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    for (int tile_base = 0; tile_base < get_global_size(0); tile_base += TILE_SIZE) {
        int tile_index = tile_base + lid;
        if (tile_index < get_global_size(0)) {
            input_tile[lid] = input[tile_index];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tile_index < get_global_size(0)) {
            output[tile_index] = silu_func(input_tile[lid]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}