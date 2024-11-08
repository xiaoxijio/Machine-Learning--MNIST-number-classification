基于MINST的数字分类

[MNIST]([Deep Learning Project - Handwritten Digit Recognition using Python - DataFlair](https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/))这可能是机器学习和深度学习爱好者中最受欢迎的数据集之一（这句话说起来好营销号啊）。MNIST[数据集](http://yann.lecun.com/exdb/mnist/)包含 60,000 张从 0 到 9 的手写数字训练图像和 10,000 张用于测试的图像。因此，MNIST 数据集有 10 个不同的类别。手写数字图像表示为 28×28 矩阵，其中每个单元包含灰度像素值。

![image](https://github.com/user-attachments/assets/e22c8949-eac6-4036-9b2b-11dbf8ad2b00)


1、导入数据集并解码

2、将数据都转为tensor

3、设置一些参数，包括批次大小、训练次数、设备选择、隐藏层、优化器之类的，这些都可以试试调成其它的看看效果

4、训练。TensorDataset将 x_train和 y_train封装为一个数据集, DataLoader将数据划分为批量, shuffle数据顺序随机。将模型设置为train，方便使用dropout 或 batch normalization 等训练特性，虽然我在这里没有用到。然后就是基本操作，将训练数据传到模型里，损失函数计算模型给出的结果与实际结果的差距，然后反向传播，梯度下降，更新参数，都是这个套路。

5、测试。就是没有训练的反向传播过程，直接带到训练好的模型里。看看效果

![image](https://github.com/user-attachments/assets/76dde03a-78bf-4c5a-88b1-a826d0a5cf11)


准确率没到达100%啊，留给大家调参了（绝对不是我懒得调）
