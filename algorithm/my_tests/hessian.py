import torch

# 模拟输入和线性层参数
a, b = 4, 3  # 输入和输出特征数
x = torch.randn(1, a, requires_grad=True)  # 输入
W = torch.randn(a, b, requires_grad=True)  # 权重
t = torch.randn(1, b)  # 目标输出

print('x', x)
print('x^2', x**2)

# 前向计算
y = x @ W  # 线性层输出
loss = 0.5 * torch.sum((y - t)**2)  # 均方误差

# 一阶梯度
grad = torch.autograd.grad(loss, W, create_graph=True)[0]
print(grad)

# 二阶 Hessian 矩阵
hessian = []
for g in grad.view(-1):  # 展平梯度以逐元素计算
    row = torch.autograd.grad(g, W, retain_graph=True)[0].view(-1)
    hessian.append(row)
hessian = torch.stack(hessian).view(a * b, a * b)

print("Hessian Matrix:")
print(hessian)

