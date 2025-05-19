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
        hidden_states = outputs.hidden_states
        # 返回倒数第二层的 hidden state（-2 表示倒数第二层）
        return outputs.logits

# 构造 dummy 输入（带 [MASK]）
text = "The capital of France is [MASK]."
dummy_input = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)

input_ids = dummy_input["input_ids"]
print(input_ids[0])
# decode the input  ids
print([tokenizer.decode(e) for e in input_ids[0]])
attention_mask = dummy_input["attention_mask"]
print(attention_mask)

# 包装后 trace
wrapped = Wrapper(model)

result = wrapped(input_ids, attention_mask)
print(result)
print(result.detach().numpy().flatten().tolist()[0])

output_token_ids = torch.argmax(result, dim=-1)
print(output_token_ids)
output_text = tokenizer.decode(output_token_ids[0])
print(output_text)