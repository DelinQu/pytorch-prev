# 随机梯度下降
import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = 1.0
 
def forward(x):
    return x*w

# calculate loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

# 定义随机梯度，不是整体的损失
# define the gradient function  sgd
def gradient(x, y):
    return 2*x*(x*w - y)

epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x,y in zip(x_data, y_data):
        # 每一样本都进行梯度计算和权值更新
        grad = gradient(x,y)
        w = w - 0.01*grad    # update weight by every grad of sample of training set
        print("\tgrad:", x, y,grad)
        l = loss(x,y)
    print("progress:",epoch,"w=",w,"loss=",l)
    epoch_list.append(epoch)
    loss_list.append(l)

# 绘图
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show() 