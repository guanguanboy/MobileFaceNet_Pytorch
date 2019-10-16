import torch
import torch.nn as nn
import torch.nn.functional as F

m = nn.Conv1d(1, 2, 3, stride=2)
input = torch.ones(1, 1, 11)

print(input.size())
print(input)
output = m(input)

print(output.size())
print(output)

# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)


m = nn.MaxPool2d(3, stride=2)
input = torch.ones(20, 16, 10, 8)
output = m(input)
print("maxpool2d output size =")
print(output.size())
print(output)