__kernel 
void gelu(__global float *input, __global float *output) 
{
  int gid = get_global_id(0);
  // GeLU激活函数: x * 0.5 * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3)))
  // 常量计算
  const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
  const float coeff = 0.044715f;
  
  float x = input[gid];
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
  float tanh_inner = tanh(inner);
  
  output[gid] = 0.5f * x * (1.0f + tanh_inner);
}