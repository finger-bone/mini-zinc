__kernel 
void sigmoid(__global float *input, __global float *output) 
{
  int gid = get_global_id(0);
  output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}