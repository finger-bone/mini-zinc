__kernel void layernorm(
    __global const float *input,
    __global float *output,
    __global const float *gamma,
    __global const float *beta,
    const int batch_size,
    const int inner_dim,    // H*W 或者 seq_len
    const int channel_size, // 例如 embed_dim
    const float eps
) {
    int gid = get_global_id(0);

    int sample_and_spatial = gid / channel_size;
    int feature_idx = gid % channel_size;

    int sample_idx  = sample_and_spatial / inner_dim;
    int spatial_idx = sample_and_spatial % inner_dim;

    // 计算 mean/variance 只在 channel_size 范围内
    int base = (sample_and_spatial)*channel_size;
    float mean = 0.f, var = 0.f;
    for (int c = 0; c < channel_size; ++c) {
        float v = input[base + c];
        mean += v;
    }
    mean /= channel_size;
    for (int c = 0; c < channel_size; ++c) {
        float diff = input[base + c] - mean;
        var += diff * diff;
    }
    var /= channel_size;

    float normalized = (input[gid] - mean) / sqrt(var + eps);
    output[gid] = normalized * gamma[feature_idx] + beta[feature_idx];
}