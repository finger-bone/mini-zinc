__kernel void rsmnorm(
    __global const float* input,
    __global float* output,
    __global const float* gamma,
    int batch,
    int inner,
    int channel,
    float eps
) {
    int gid = get_global_id(0);
    int i = gid / channel;
    int j = gid % channel;

    if (i >= batch * inner) return;

    int base = i * channel;

    // 1. 计算 RMS
    float mean_square = 0.0f;
    for (int k = 0; k < channel; ++k) {
        float val = input[base + k];
        mean_square += val * val;
    }
    mean_square /= (float)channel;
    float denom = sqrt(mean_square + eps);

    // 2. 归一化 + 乘 gamma
    output[gid] = (input[gid] / denom) * gamma[j];
}
