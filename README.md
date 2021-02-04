## Pytorch实践课程

> 官网：https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#id14



### 第1讲——线性回归

- 在训练过程中绘制图像

使用 `visdom`：https://github.com/fossasia/visdom



### 第2讲——梯度下降法

- 使用随机梯度下降

- 性能和复杂度折中，小批量方式mini-batch

- 训练失败可能是学习率太大
- 一般我们选择随机梯度下降



### 第3讲——BP

- 矩阵的求导：http://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf

- 要及时对梯度清零



### 第4讲——用Pytorch实现线性回归

- 不同优化器的作用：https://blog.csdn.net/weixin_44841652/article/details/105068509
- 四步骤

![image-20210204102329528](https://i.loli.net/2021/02/04/CX8oJKzT7QvbMFe.png)

- 使用模板

### 第5讲 ——逻辑斯蒂回归

- 分类问题的思路

```
离散，求出每一个类的概率，然后取其中分类概率最大的类属作为分类结果
```

- 几类sigmod函数

![image-20210204104257546](https://i.loli.net/2021/02/04/PzJdCpqlWByIkij.png)

- 逻辑回归需要新增激活函数

![image-20210204110144950](https://i.loli.net/2021/02/04/EIWzsM4U9QlKSC6.png)

- 相应损失函数书的计算有所变化

![image-20210204110210074](https://i.loli.net/2021/02/04/bzfuClaqQsMWe9V.png)

- BCELoss - Binary CrossEntropyLoss （二维）

  ```
  BCELoss 是CrossEntropyLoss的一个特例，只用于二分类问题，而CrossEntropyLoss可以用于二分类，也可以用于多分类。
  如果是二分类问题，建议BCELoss
  ```

  