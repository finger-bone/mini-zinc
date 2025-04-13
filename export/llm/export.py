import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 选择模型和 tokenizer
model_id = "./SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

# 包装一层，避免多余参数干扰
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # 只保留主要的 forward 输出
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

# 构造 dummy 输入
dummy_input = tokenizer("Hello world, this is a piece of source text with 16 token." * 16 * 2, return_tensors="pt", padding=True, add_special_tokens=False)
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]

# 包装后 trace
wrapped = Wrapper(model)
traced = torch.jit.trace(wrapped, (input_ids, attention_mask))
traced.save("SmolLM2-135M-Instruct.pt")
print("✅ Saved TorchScript model to SmolLM2-135M-Instruct.pt")