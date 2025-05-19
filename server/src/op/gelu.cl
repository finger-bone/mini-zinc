inline float erf_approx(float x) {
    float a1 =  0.254829592f;
    float a2 = -0.284496736f;
    float a3 =  1.421413741f;
    float a4 = -1.453152027f;
    float a5 =  1.061405429f;
    float p  =  0.3275911f;

    int sign = x < 0.0f ? -1 : 1;
    x = fabs(x);

    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * exp(-x * x));

    return sign * y;
}

__kernel 
void gelu(__global float *input, __global float *output) 
{
    int gid = get_global_id(0);
    float x = input[gid];

    float sqrt_2 = 1.4142135623730951f;
    float val = 0.5f * x * (1.0f + erf_approx(x / sqrt_2));
    output[gid] = val;
}