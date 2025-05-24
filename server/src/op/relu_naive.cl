__kernel 
void relu(__global float *input, __global float *output, const float threshold) 
{
  int gid = get_global_id(0);
  if (input[gid] > threshold) {
    output[gid] = input[gid];
  } else {
    output[gid] = 0.0f;
  }
}
