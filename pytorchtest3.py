#演示view与size函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy    as np

x = torch.randn(4,4)
print(x.size())

y = x.view(16)
print(y.size())

z = x.view(-1,8)
print(z.size())

a=np.array([[[1,2,3],[4,5,6]]])


unpermuted=torch.tensor(a)
print(unpermuted.size())  #  ——>  torch.Size([1, 2, 3])
print(unpermuted)

permuted=unpermuted.permute(2,0,1)
print(permuted.size())
print(permuted)