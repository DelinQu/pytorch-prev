import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#这里设函数为y=3x+2
x_data = [1.0,2.0,3.0]
y_data = [5.0,8.0,11.0]

# 前馈
def forward(x):
    return x * w + b
# 损失
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

# 记录训练参数
mse_list = []
W=np.arange(0.0,4.1,0.1)
B=np.arange(0.0,4.1,0.1)

# 生成网格采样点
[w,b]=np.meshgrid(W,B)

# 训练
l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)     # 自动计算了整个列表，不用循环
    print(y_pred_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

# 绘图
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(w, b, l_sum/3)
plt.show()
