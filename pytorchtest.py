import torch

x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y * 3

out = z.mean()

print(z, out)

out.backward()

print(x.grad)

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) #返回一个新张量，它的数据与 self 张量相同，但 shape 不同， -1表示该维的大小需要从其它推断出来
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

#测试上面定义的网络
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

#将所有参数的梯度缓存清零，然后进行随机梯度的的反向传播
net.zero_grad()
out.backward(torch.randn(1, 10))

#损失函数：一个损失函数接受一对 (output, target) 作为输入，计算一个值来估计网络的输出和目标值相差多少。
output = net(input)
target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使target和output的shape相同
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

#反向传播：调用loss.backward()获得反向传播的误差。但是在调用前需要清除已存在的梯度，否则梯度将被累加到已存在的梯度。
#现在，我们将调用loss.backward()，并查看conv1层的偏差（bias）项在反向传播前后的梯度。
net.zero_grad()     # 清除梯度

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
