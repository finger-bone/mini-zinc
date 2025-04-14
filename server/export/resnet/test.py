# export_resnet.py
import torch
import torchvision.models as models
import time

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.model(x)

model = Wrapper().to('mps')
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).to('mps')

start = time.time()
with torch.no_grad():
    res = model(dummy_input)
end = time.time()

print(res)

print(f"ms: {(end - start) * 1000:.3f} ms")