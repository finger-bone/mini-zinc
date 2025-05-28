__kernel void softmax_safe(
    __global float* input,
    __global float* output,
    __global float* maxval,
    int batch,
    int n
) {
    int b = get_global_id(0);
    if (b >= batch) return;
    int base = b * n;
    float maxv = maxval[b];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = exp(input[base + i] - maxv);
        sum += e;
    }
    for (int i = 0; i < n; ++i) {
        float e = exp(input[base + i] - maxv);
        output[base + i] = e / sum;
    }
}