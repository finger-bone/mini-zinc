__kernel 
void batchnorm(__global float *input, __global float *output, __global float *mean, __global float *var, __global float *gamma, __global float *beta, const float epsilon, const int channels, const int spatial_size) 
{
  int gid = get_global_id(0);
  int n = gid / (channels * spatial_size);
  int c = (gid / spatial_size) % channels;
  
  // 应用批归一化公式: y = gamma * (x - mean) / sqrt(var + epsilon) + beta
  float normalized = (input[gid] - mean[c]) / sqrt(var[c] + epsilon);
  output[gid] = gamma[c] * normalized + beta[c];
}