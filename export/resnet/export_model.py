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

dummy_input = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(model, dummy_input)

# 保存为 torchscript 文件
traced.save("resnet18.pt")