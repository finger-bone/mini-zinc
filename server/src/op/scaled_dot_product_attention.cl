__kernel void scaled_dot_product_attention(
    __global float* q,
    __global float* k,
    __global float* v,
    __global float* mask,
    __global float* output,
    __global float* temp_buffer, // 临时缓冲区用于存储logits
    int batch,
    int heads,
    int seq_len,
    int embed_dim,
    float dropout,
    float scale,
    int has_mask
) {
    int global_id = get_global_id(0);
    int total = batch * heads * seq_len;
    if (global_id >= total) return;

    // 计算位置索引
    int b = global_id / (heads * seq_len);
    int h = (global_id / seq_len) % heads;
    int i = global_id % seq_len; // 当前query的位置

    // QKV偏移计算
    int qkv_offset = ((b * heads + h) * seq_len + i) * embed_dim;
    int logits_offset = (b * heads * seq_len + h * seq_len + i) * seq_len;

    // 计算 attention logits
    float max_logit = -FLT_MAX;

    for (int j = 0; j < seq_len; ++j) {
        float dot = 0.0f;
        int k_offset = ((b * heads + h) * seq_len + j) * embed_dim;
        for (int d = 0; d < embed_dim; ++d) {
            dot += q[qkv_offset + d] * k[k_offset + d];
        }
        dot *= scale;

        // 加mask
        if (has_mask) {
            // int mask_offset = ((b * heads + h) * seq_len + i) * seq_len + j;
            // int mask_offset = b * seq_len + i;
            int mask_offset = (b * seq_len + i) * seq_len + j;
            // if (mask[mask_offset] == 1.0f) {
            //     dot = -FLT_MAX;
            // }
            dot += mask[mask_offset];
        }

        temp_buffer[logits_offset + j] = dot;
        if (dot > max_logit) max_logit = dot;
    }

    // softmax with numerical stability
    float sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float logit = temp_buffer[logits_offset + j];
        float exp_val = exp(logit - max_logit);
        temp_buffer[logits_offset + j] = exp_val;
        sum += exp_val;
    }

    // Normalize softmax probabilities
    float sum_reciprocal = 1.0f / (sum + 1e-9f); // 添加小值防止除零
    for (int j = 0; j < seq_len; ++j) {
        temp_buffer[logits_offset + j] *= sum_reciprocal;
    }

    // attention × V
    for (int d = 0; d < embed_dim; ++d) {
        float val = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            int v_offset = ((b * heads + h) * seq_len + j) * embed_dim + d;
            val += temp_buffer[logits_offset + j] * v[v_offset];
        }

        int out_offset = ((b * heads + h) * seq_len + i) * embed_dim + d;
        output[out_offset] = val;
    }
}