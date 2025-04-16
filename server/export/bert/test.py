import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 使用支持填空任务的模型
model_id = "./distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)
model.eval()

# 包装器，只返回 logits（每个 token 的预测分布）
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

# 构造 dummy 输入（带 [MASK]）
text = "The capital of France is <mask>."
dummy_input = tokenizer(text, return_tensors="pt", padding="max_length", max_length=482)

input_ids = dummy_input["input_ids"]
print(len(input_ids[0]))
attention_mask = dummy_input["attention_mask"]
print(attention_mask)

# 包装后 trace
wrapped = Wrapper(model)

result = wrapped(input_ids, attention_mask)
print(result)
