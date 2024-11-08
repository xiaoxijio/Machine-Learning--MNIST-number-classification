import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


class CNN(nn.Module):
    """
    这次跟上次不一样了，这次是直接将图片放模型里(1,28,28)，上次是把图像数据平铺放里面的(28 * 28 = 784)
    图像大小 (1,28,28) 注: pytorch中, 图像张量的格式通常是 (C, H, W),  OpenCV中，通常使用 (H, W, C) 格式
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 因为是灰度图, 所以通道为 1
                out_channels=16,  # 输出通道数，决定了该层生成的特征图数量
                kernel_size=5,  # 卷积核大小 (通常越小捕捉的细节越好, 而且所需的参数也少很多, 通常会设为 3或 5)
                stride=1,  # 步长
                padding=2  # 边缘填充
            ),
            nn.ReLU(),  # 激活！
            nn.MaxPool2d(kernel_size=2)  # 最大池化层
        )
        self.conv2 = nn.Sequential(  # 记得输入要对应上面的输出
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 为了更深入了解, 大家可以自己计算一下每层输出数据的size ,这里为(32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def train(train_steps, train_batch, model, opt, device):
    print('训练开始')
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for step in range(train_steps):
        for i, (batch_x, batch_y) in enumerate(train_batch):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predict_y = model(batch_x)
            loss = loss_fn(predict_y, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if step % 10 == 0:
            print('train_loss:', loss.item())  # 简单看看 不搞求和平均了
    print('训练完成')


@torch.no_grad()
def test(test_batch, model, device):
    print('测试开始')
    model.eval()
    correct = 0
    total = 0
    for i, (batch_x, batch_y) in enumerate(test_batch):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        predict_y = model(batch_x)
        predicted = torch.max(predict_y, 1)[1]
        correct += (predicted == batch_y).sum().item()  # 求和预测对的
        total += batch_y.size(0)

    accuracy = correct / total
    print(f"测试准确率: {accuracy * 100:.2f}%")


batch_size = 64
train_steps = 100
# 内置的 MNIST数据集 如果./data路径上有这个数据集，将不会再下载 (**之前不知道 还去网上找  真**了 文明用语)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
# image, label = train_dataset[0]  # 可以看一下数据
# plt.imshow(image.squeeze(), cmap="gray")
# plt.show()
# print(image.size())
train_batch = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu还是cpu
model = CNN().to(device)  # 模型
opt = optim.Adam(model.parameters(), lr=0.001)  # 优化器

train(train_steps, train_batch, model, opt, device)  # 训练
test(test_batch, model, device)  # 测试
