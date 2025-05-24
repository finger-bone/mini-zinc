import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# 设置设备为 MPS（macOS 上的 GPU）
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 模型目录为 CausalLM 类型
model_id = "./SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval().to(device)

# 包装器，只输出 logits
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# 构造输入（没有 [MASK]）
text = "The capital of China is"
dummy_input = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32, truncation=True)

input_ids = dummy_input["input_ids"].to(device)
attention_mask = dummy_input["attention_mask"].to(device)

# 推理 + 时间统计
start = time.time()
wrapped = Wrapper(model).to(device)
logits = wrapped(input_ids, attention_mask)
end = time.time()

# 获取最后一个位置的预测 token
last_token_index = attention_mask.sum(dim=1) - 1
next_token_logits = logits[0, last_token_index, :]  # shape: [vocab_size]
predicted_token_id = torch.argmax(next_token_logits).item()
predicted_word = tokenizer.decode([predicted_token_id])

# 解码整个生成的序列（也可以选择用 generate）
output_token_ids = torch.cat([input_ids[0], torch.tensor([predicted_token_id]).to(device)], dim=0)
output_text = tokenizer.decode(output_token_ids, skip_special_tokens=True)

# 输出信息
print(f"🕙 推理时间: {(end - start) * 1000:.2f} ms")
print("👉 原始输入:", text)
print("👉 预测 token:", predicted_word)
print("👉 拼接后输出:", output_text)
