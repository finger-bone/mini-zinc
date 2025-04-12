__kernel 
void softmax(__global float *input, __global float *output, const int size, const int axis_size) 
{
  int gid = get_global_id(0);
  int batch_idx = gid / axis_size;
  int start_idx = batch_idx * axis_size;
  
  // 找到最大值以提高数值稳定性
  float max_val = input[start_idx];
  for (int i = 0; i < axis_size; i++) {
    max_val = max(max_val, input[start_idx + i]);
  }
  
  // 计算指数和
  float sum = 0.0f;
  for (int i = 0; i < axis_size; i++) {
    sum += exp(input[start_idx + i] - max_val);
  }
  
  // 计算softmax
  int offset = gid % axis_size;
  output[gid] = exp(input[gid] - max_val) / sum;
}