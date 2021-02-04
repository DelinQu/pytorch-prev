# BP反向传播
import torch

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 定义torch
w = torch.Tensor([1.0]) # w的初始值为1.0
w.requires_grad = True  # 需要的计算梯度

# 前馈和损失的计算实际上就构建了计算图
# 前馈
def forward(x):
    return x*w          # w是一个Tensor

# 损失
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

# 计算训练前的预测值（输出）
print("predict (before training)", 4, forward(4).item())

# 开始训练
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)   # l是一个张量，tensor主要是在建立计算图 forward   
        l.backward()    # 使用backward自动计算计算图的反向传播
        w.data = w.data - 0.01*w.grad.data  # 更新权重

        # 更新权重之后要清零梯度
        w.grad.data.zero_()
    print('progress:', epoch, l.item())

# 预测
print("predict (after training)", 4, forward(4).item())
