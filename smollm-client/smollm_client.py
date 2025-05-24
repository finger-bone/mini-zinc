from transformers import AutoTokenizer
import requests
from sys import argv
import torch

tokenizer = AutoTokenizer.from_pretrained("../server/export/smollm/SmolLM2-135M-Instruct")

text = "The capital of China is"
tokenized_input = tokenizer(text, return_tensors="pt", padding="max_length", max_length=32)
tokenized_input_no_padding = tokenizer(text, return_tensors="pt")

input_ids_no_padding = tokenized_input_no_padding["input_ids"]

input_ids = tokenized_input["input_ids"]
# print the decoded input ids
print([tokenizer.decode(e) for e in input_ids[0]])

attention_mask = tokenized_input["attention_mask"]

request_data = {
    "inputs": {
        "0": {
            "dtype": "int64",
            "shape": input_ids.shape,
            "data_i64": input_ids.flatten().tolist()
        },
        "1": {
            "dtype": "int64",
            "shape": attention_mask.shape,
            "data_i64": attention_mask.flatten().tolist()
        }
    }
}
# print(request_data)


response = requests.post("http://127.0.0.1:3030/infer", json=request_data)
resp = response.json()
print(f"🕙推理时间: {resp['duration_ms']}ms")
# print(resp["outputs"].keys())
logits = resp["outputs"]["1421"]["data_f32"]
shape = resp["outputs"]["1421"]["shape"]
logits = torch.tensor(logits).reshape(shape)
print(logits.shape)
# decode the logits, output the whole output sentence
output_ids = logits.argmax(dim=-1).tolist()
print(output_ids)
output_text = tokenizer.decode(output_ids[0][len(input_ids_no_padding[0]) - 1], skip_special_tokens=False)
print(output_text)
