基于MINST的数字分类

[MNIST]([Deep Learning Project - Handwritten Digit Recognition using Python - DataFlair](https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/))这可能是机器学习和深度学习爱好者中最受欢迎的数据集之一（这句话说起来好营销号啊）。MNIST[数据集](http://yann.lecun.com/exdb/mnist/)包含 60,000 张从 0 到 9 的手写数字训练图像和 10,000 张用于测试的图像。因此，MNIST 数据集有 10 个不同的类别。手写数字图像表示为 28×28 矩阵，其中每个单元包含灰度像素值。

![image](https://github.com/user-attachments/assets/e22c8949-eac6-4036-9b2b-11dbf8ad2b00)

我用了两种方法，一种是将数据平铺放全连接层里训练(28 * 28 = 784)，一种是直接拿图像用卷积训练(1,28,28)。
大家可以先看数字分类再看卷积数字分类，数字分类里有些步骤我拆解下来一一分析，卷积同样的操作我就废话不多说了。  
1、导入数据集并解码

2、将数据都转为tensor

3、设置一些参数，包括批次大小、训练次数、设备选择、隐藏层、优化器之类的，这些都可以试试调成其它的看看效果

4、训练。TensorDataset将 x_train和 y_train封装为一个数据集, DataLoader将数据划分为批量, shuffle数据顺序随机。将模型设置为train，方便使用dropout 或 batch normalization 等训练特性，虽然我在这里没有用到。然后就是基本操作，将训练数据传到模型里，损失函数计算模型给出的结果与实际结果的差距，然后反向传播，梯度下降，更新参数，都是这个套路。

5、测试。就是没有训练的反向传播过程，直接带到训练好的模型里。看看效果（第一张是数字分类，第二张时卷积数字分类）

![image](https://github.com/user-attachments/assets/76dde03a-78bf-4c5a-88b1-a826d0a5cf11)
![image](https://github.com/user-attachments/assets/ac00b8fc-b102-4d71-b220-30802a5d6fa1)



这两个效果不一样但是不要做对比哦，因为两个我用的优化器和损失函数，还有运行的时间都不一样，大家可以自己调整对比（这种小型的数据集训练出来的效果其实都差不多）
准确率没到达100%啊，留给大家调参了（绝对不是我懒得调）
