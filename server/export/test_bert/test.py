import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=4, d_model=8, num_heads=2, max_len=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn1 = nn.Linear(d_model, d_model * 4)
        self.ffn2 = nn.Linear(d_model * 4, d_model)

        self._init_weights()

    def _init_weights(self):
        torch.manual_seed(42)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask=None):
        """
        x: LongTensor [B, T]
        attention_mask: Bool or Byte Tensor [B, T], True for valid tokens, False for padding.
        """
        B, T = x.shape

        x = self.embed(x)  # [B, T, D]
        x = x + self.pos_embedding[:, :T, :]

        # Attention sublayer
        residual = x
        x = self.ln1(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def reshape_heads(tensor):
            B, T, D = tensor.size()
            tensor = tensor.view(B, T, self.num_heads, self.head_dim)
            return tensor.transpose(1, 2)  # [B, nh, T, hd]

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        # 处理attention_mask，生成attn_mask给scaled_dot_product_attention
        if attention_mask is not None:
            # attention_mask: [B, T] bool or byte
            # 转成float，True->0, False->-inf，形状扩展到[ B, 1, 1, T ]，用于key的mask
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
            attn_mask = attn_mask.to(dtype=torch.float32)
            attn_mask = (1.0 - attn_mask) * -1e9  # padding位置是 -inf，其它是0
        else:
            attn_mask = None
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        print(attn_mask)
        return attn_out
        x = self.out_proj(attn_out)
        x = residual + x

        # FFN sublayer
        residual = x
        x = self.ln2(x)
        x = self.ffn1(x)
        x = F.gelu(x)
        x = self.ffn2(x)
        x = residual + x



# 构造模型和 dummy 输入
model = SimpleTransformer()
model.eval()
input_ids = torch.tensor([
    [1, 0, 1, 2]
])

attention_mask = (input_ids != 0)  # padding是0的地方mask False，其他True

output = model(input_ids, attention_mask=attention_mask)
print(output.shape)  # [2, 6, 8]
print(output)