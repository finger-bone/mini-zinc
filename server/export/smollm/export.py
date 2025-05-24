import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 替换为 CausalLM 类型模型
model_id = "./SmolLM2-135M-Instruct"  # 修改为你的模型目录
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

# 包装器：仅输出 logits
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits  # [batch_size, seq_len, vocab_size]

# 构造 dummy 输入（普通文本）
text = "The capital of China is"
dummy_input = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]

# 包装并 trace
wrapped = Wrapper(model)
traced = torch.jit.trace(wrapped, (input_ids, attention_mask))
traced.save("smollm.pt")

print("✅ Saved TorchScript model for CausalLM to causal-lm-traced.pt")