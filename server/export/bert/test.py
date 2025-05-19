import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 设置设备为 MPS（Mac 上的 GPU 加速）
device = "mps"

model_id = "./distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
model.eval().to(device)  # 模型迁移到 MPS

# 包装器，只返回 logits（每个 token 的预测分布）
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        return outputs.logits

# 构造 dummy 输入（带 [MASK]）
text = "The capital of china is [MASK]."
dummy_input = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)

# 将输入迁移到 MPS

import time
start = time.time()
input_ids = dummy_input["input_ids"].to(device)
attention_mask = dummy_input["attention_mask"].to(device)

wrapped = Wrapper(model).to(device)  # 包装器也迁移到 MPS
result = wrapped(input_ids, attention_mask)
output_token_ids = torch.argmax(result, dim=-1)
output_text = tokenizer.decode(output_token_ids[0].cpu())  # decode 前迁回 CPU
end = time.time()
print(f"🕙模型推理时间: {(end - start) * 1000}ms")


print(input_ids[0])
print([tokenizer.decode(e) for e in input_ids[0]])
print(attention_mask)
print(result)
print(result.detach().cpu().numpy().flatten().tolist()[0])  # detach 后也迁回 CPU
print(output_token_ids)
print(output_text)
