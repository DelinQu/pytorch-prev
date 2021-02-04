import torch
input = [3,4,6,5,7,
        2,4,6,8,2,
        1,6,7,8,4,
        9,7,4,6,2,
        3,7,5,4,1]

# 使用view固定为一个张量B×C×W×H
input = torch.Tensor(input).view(1, 1, 5, 5)

# 定义卷积
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)

# 将一个向量转化为张量赋值给kernel卷积核 
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)

# 卷积权重的初始化
conv_layer.weight.data = kernel.data

# 卷积
output = conv_layer(input)
print(output)
