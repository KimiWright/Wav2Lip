import torch
import torch.nn as nn

m = nn.AdaptiveAvgPool2d((2, 2))

input = torch.randn(1, 3 ,3)
print(input)
output = m(input)
print(output)
print(output.shape) 