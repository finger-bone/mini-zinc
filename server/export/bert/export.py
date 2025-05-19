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
        # 开启输出所有 hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # 关键参数
        )
        # 获取所有层的 hidden states（列表）
        # hidden_states = outputs.hidden_states
        # return hidden_states[1]
        return outputs.logits

# 构造 dummy 输入（带 [MASK]）
text = "The capital of France is <mask>."
dummy_input = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)
input_ids = dummy_input["input_ids"]
print(len([tokenizer.decode(e) for e in input_ids[0]]))
attention_mask = dummy_input["attention_mask"]

# 包装后 trace
wrapped = Wrapper(model)
traced = torch.jit.trace(wrapped, (input_ids, attention_mask))
traced.save("distilbert-base-uncased-base-fillmask.pt")

print("✅ Saved TorchScript model for fill-mask to distilbert-base-uncased-base-fillmask.pt")
