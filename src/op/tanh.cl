__kernel 
void tanh(__global float *input, __global float *output) 
{
  int gid = get_global_id(0);
  output[gid] = tanh(input[gid]);
}