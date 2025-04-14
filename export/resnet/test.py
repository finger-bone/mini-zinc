# export_resnet.py
import torch
import torchvision.models as models

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

    def forward(self, x):
        return self.model(x)

# 用 torchscript trace
model = Wrapper()
model.eval()

dummy_input = torch.zeros(1, 3, 224, 224)
print(model.forward(dummy_input))