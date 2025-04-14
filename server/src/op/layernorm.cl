// 补全层归一化计算逻辑
__kernel void layernorm(
    __global float* input,
    __global float* output,
    __global float* gamma,
    __global float* beta,
    const float eps,
    const int embed_dim,
    const int batch_seq // batch_size * seq_len
) {
    int gid = get_global_id(0);
    int sample_idx = gid / embed_dim;
    int embed_idx = gid % embed_dim;

    // 计算当前样本的均值和方差
    float mean = 0.0f;
    for (int i = 0; i < embed_dim; i++) {
        mean += input[sample_idx * embed_dim + i];
    }
    mean /= embed_dim;

    float var_sum = 0.0f;
    for (int i = 0; i < embed_dim; i++) {
        float diff = input[sample_idx * embed_dim + i] - mean;
        var_sum += diff * diff;
    }
    float var = rsqrt( (var_sum / embed_dim) + eps );

    // 应用层归一化公式
    float x = input[sample_idx * embed_dim + embed_idx];
    output[gid] = (x - mean) * var * gamma[embed_idx] + beta[embed_idx];
}