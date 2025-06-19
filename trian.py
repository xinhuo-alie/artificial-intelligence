import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
import cv2

torch.manual_seed(1)  # 设置随机种子保证结果可复现

# 超参数
EPOCH = 5  # 增加训练轮次提高准确率
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True if not os.path.exists('./data/MNIST/') else False

# 下载并加载MNIST数据集
train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False
)

# 数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 准备测试数据
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]  # 注意：已从test_labels重命名为targets


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

# 优化器和损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练模型
if not os.path.exists('cnn2.pkl'):  # 检查模型文件是否存在
    print("开始训练模型...")
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = (pred_y == test_y.numpy()).mean()
                print(f'Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f} | Test Acc: {accuracy:.4f}')

    # 保存模型
    torch.save(cnn.state_dict(), 'cnn2.pkl')
    print("模型已保存为 cnn2.pkl")
else:
    print("加载已存在的模型...")
    cnn.load_state_dict(torch.load('cnn2.pkl'))

cnn.eval()

# 测试模型
print("进行模型测试...")
inputs = test_x[:32]
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print("预测数字:", pred_y)
print("真实数字:", test_y[:32].numpy())

# 可视化测试图像
img = torchvision.utils.make_grid(inputs)
img = img.numpy().transpose(1, 2, 0)

# 显示图像
cv2.imshow('Test Images with Predictions', img)
print(f"按任意键退出...")
cv2.waitKey(0)
cv2.destroyAllWindows()