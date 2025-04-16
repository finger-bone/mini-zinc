__kernel void layernorm(
    __global const float *input,
    __global float *output,
    __global const float *gamma,
    __global const float *beta,
    const int batch_size,
    const int feature_size,
    const float eps
) {
    int global_id = get_global_id(0);
    int batch_idx = global_id / feature_size;
    int feature_idx = global_id % feature_size;

    // Calculate mean and variance for the current batch element
    float mean = 0.0f;
    float variance = 0.0f;
    int offset = batch_idx * feature_size;

    for (int i = 0; i < feature_size; ++i) {
        mean += input[offset + i];
    }
    mean /= feature_size;

    for (int i = 0; i < feature_size; ++i) {
        float diff = input[offset + i] - mean;
        variance += diff * diff;
    }
    variance /= feature_size;

    // Normalize
    float normalized_val = (input[global_id] - mean) / sqrt(variance + eps);

    // Scale and shift
    output[global_id] = normalized_val * gamma[feature_idx] + beta[feature_idx];
}