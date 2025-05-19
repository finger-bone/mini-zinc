__kernel void masked_fill(
    __global const float* input,
    __global const uchar* mask,
    __global float* output,
    const float value,
    const unsigned long size
) {
    size_t idx = get_global_id(0);
    if (idx < size) {
        output[idx] = mask[idx] <= 0.1f ? input[idx] : value;
    }
}