// inline float erf_approx(float x) {
//     float a1 =  0.254829592f;
//     float a2 = -0.284496736f;
//     float a3 =  1.421413741f;
//     float a4 = -1.453152027f;
//     float a5 =  1.061405429f;
//     float p  =  0.3275911f;

//     int sign = x < 0.0f ? -1 : 1;
//     x = fabs(x);

//     float t = 1.0f / (1.0f + p * x);
//     float y = 1.0f - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-x * x));

//     return sign * y;
// }

// inline float gelu_approx(float x) {
//     float sqrt_2 = 1.4142135623730951f;
//     float val = 0.5f * x * (1.0f + erf_approx(x / sqrt_2));
//     return val;
// }

inline float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f; // ≈ sqrt(2/pi)
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

__kernel 
void gelu(__global float *input, __global float *output) 
{
    int gid = get_global_id(0);
    float x = input[gid];
    float val = gelu_approx(x);
    output[gid] = val;
}