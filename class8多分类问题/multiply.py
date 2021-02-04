import torch
# 用于数据集的加载
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# 使用激活函数relu
import torch.nn.functional as F
# 优化器
import torch.optim as optim

# 准备数据集
batch_size = 64
# 归一化,均值和方差
transform = transforms.Compose([transforms.ToTensor(),  # Convert the PIL Image to Tensor.
                                transforms.Normalize((0.1307,), (0.3081,))]) 

train_dataset = datasets.MNIST(root='../data/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='../data/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 设计模型类
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 5 层
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
 
    # 前馈
    def forward(self, x):
        x = x.view(-1, 784)     # -1其实就是自动获取mini_batch,使用wiew展开张量为向量
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)       # 最后一层不做激活，不进行非线性变换
 
 
model = Net()
 
# 构造损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()                             # 交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)    # SGD优化器

# 训练
# training cycle forward, backward, update 
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)     
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # 每300个数据打印一次损失率
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
# 预测
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # 预测 结果为一个batch长度×Feature个数的张量
            outputs = model(images)
            # torch.max的返回值有两个，第一个是每一行的最大值是多少，第二个是每一行最大值的下标(索引)是多少。
            _, predicted = torch.max(outputs.data, dim=1) # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()