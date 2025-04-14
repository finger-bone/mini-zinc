# export_resnet.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

        # 参数初始化为全 1
        nn.init.constant_(self.conv.weight, 1.0)
        nn.init.constant_(self.conv.bias, 1.0)
        nn.init.constant_(self.fc.weight, 1.0)
        nn.init.constant_(self.fc.bias, 1.0)

    def forward(self, x):
        x = self.conv(x)       # -> [B, 16, H, W]
        x = self.relu(x)       # -> [B, 16, H, W]
        x = self.pool(x)       # -> [B, 16, 1, 1]
        x = torch.flatten(x, 1) # -> [B, 16]
        x = self.fc(x)         # -> [B, 1000]
        return x
# 用 torchscript trace
model = Wrapper()
model.eval()

dummy_input = torch.zeros(1, 3, 224, 224)
print(model.forward(dummy_input))