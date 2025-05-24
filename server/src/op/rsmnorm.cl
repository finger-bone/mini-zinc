__kernel void rsmnorm(
    __global const float *input,
    __global float *output,
    __global const float *gamma,
    __global const float *beta,
    const int batch_size,
    const int inner_dim,
    const int channel_size,
    const float eps
) {
    int gid = get_global_id(0);
    int sample_and_spatial = gid / channel_size;
    int feature_idx = gid % channel_size;
    int base = (sample_and_spatial)*channel_size;
    float mean = 0.f, var = 0.f;
    // RSMNorm: 均值和方差计算仅在非负元素上
    int count = 0;
    for (int c = 0; c < channel_size; ++c) {
        float v = input[base + c];
        if (v >= 0.0f) {
            mean += v;
            count++;
        }
    }
    mean = count > 0 ? mean / count : 0.0f;
    for (int c = 0; c < channel_size; ++c) {
        float v = input[base + c];
        if (v >= 0.0f) {
            float diff = v - mean;
            var += diff * diff;
        }
    }
    var = count > 0 ? var / count : 0.0f;
    float normalized = count > 0 ? (input[gid] - mean) / sqrt(var + eps) : 0.0f;
    output[gid] = normalized * gamma[feature_idx] + beta[feature_idx];
}