import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from unit import parse_mnist
from torch.utils.data import TensorDataset, DataLoader


class Mnist_NN(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super(Mnist_NN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.output = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


def train(train_steps, x_train, y_train, batch_size, model, opt, device):
    print('训练开始')  # TensorDataset将 x_train和 y_train封装为一个数据集, DataLoader将数据划分为批量, shuffle数据顺序随机
    model.train()  # 将模型设置为训练模式
    batch_xy = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    for step in range(train_steps):
        for batch_x, batch_y in batch_xy:
            batch_x = batch_x.view(batch_x.size(0), -1)  # [64, 28, 28] --> [62, 784]
            batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)  # torch.uint8 --> torch.float32
            loss = F.cross_entropy(model(batch_x), batch_y)  # 大家可以试试其它损失函数, 比如...忘了
            loss.backward()
            opt.step()
            opt.zero_grad()

        if step % 10 == 0:
            print('train_loss:', loss.item())  # 简单看看 不搞求和平均了
    print('训练完成')


@torch.no_grad()  # 禁用梯度计算
def test(x_test, y_test, batch_size, device):
    print('测试开始')
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    for batch_x, batch_y in DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False):
        batch_x, batch_y = batch_x.to(device).float(), batch_y.to(device)
        batch_x = batch_x.view(batch_x.size(0), -1)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()  # 求和预测对的
        total += batch_y.size(0)

    accuracy = correct / total
    print(f"测试准确率: {accuracy * 100:.2f}%")


# 对 MNIST 数据集解码 图片大小(28 * 28)
x_train = parse_mnist(minst_file_addr="data/MNIST/train-images-idx3-ubyte.gz")  # 60000份
y_train = parse_mnist(minst_file_addr="data/MNIST/train-labels-idx1-ubyte.gz")
x_test = parse_mnist(minst_file_addr="data/MNIST/t10k-images-idx3-ubyte.gz")  # 10000份
y_test = parse_mnist(minst_file_addr="data/MNIST/t10k-labels-idx1-ubyte.gz")
# plt.imshow(x_train[0])
# plt.show()

x_train, y_train, x_test, y_test = map(torch.tensor, (x_train, y_train, x_test, y_test))  # 转为 tensor
batch_size = 64  # 批次大小
train_steps = 100  # 训练次数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # gpu还是cpu
input_dim, hidden1_dim, hidden2_dim, output_dim = 784, 256, 64, 10  # 28 * 28 = 784
model = Mnist_NN(input_dim, hidden1_dim, hidden2_dim, output_dim).to(device)  # 模型
opt = optim.SGD(model.parameters(), lr=0.001)  # 优化器(大家也可以用其他优化器, 比如现在主流的Adma)
train(train_steps, x_train, y_train, batch_size, model, opt, device)  # 训练
test(x_test, y_test, batch_size, device)  # 测试
