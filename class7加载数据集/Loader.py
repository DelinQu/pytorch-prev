# 数据加载
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
 
# 准备数据集
class DiabetesDataset(Dataset):                         # 抽象类DataSet
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]                          # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])      
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
 
    def __len__(self):
        return self.len
 
# dataset对象
dataset = DiabetesDataset('../data/diabetes.csv')

# 使用DataLoader加载数据
train_loader = DataLoader(dataset=dataset,  # dataSet对象 
                            batch_size=32,  # 每个batch的条数
                            shuffle=True,   # 是否打乱
                            num_workers=4)  # 多线程一般设置为4和8
 

# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
 
 
model = Model()
 
# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
# training cycle forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            inputs, labels = data                   # 取出一个batch 
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            # 更新
            optimizer.step()