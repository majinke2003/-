# 加载一些基础的库
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm  # 一个实现进度条的库

transform = transforms.Compose({
    # 转化为Tensor
    transforms.ToTensor()
})


# 首先继承Dataset写一个对于数据进行读入和处理的方式
class MyDataset(Dataset):
    def __init__(self, path):
        self.mode = ('train' if 'mask' in os.listdir(path) else 'test')  # 表示训练模式
        self.path = path  # 图片路径
        dirlist = os.listdir(path + 'image/')  # 图片的名称
        self.name = [n for n in dirlist if n[-3:] == 'png']  # 只读取图片

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):  # 获取数据的处理方式
        name = self.name[index]
        # 读取原始图片和标签
        if self.mode == 'train':  # 训练模式
            ori_img = cv2.imread(self.path + 'image/' + name)  # 原始图片
            lb_img = cv2.imread(self.path + 'mask/' + name)  # 标签图片
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            lb_img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2GRAY)  # 转为灰度图
            return transform(ori_img), transform(lb_img)

        if self.mode == 'test':  # 测试模式
            ori_img = cv2.imread(self.path + 'image/' + name)  # 原始图片
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 转为RGB三通道图
            return transform(ori_img)


# 加载数据集
train_path = 'train/'
traindata = MyDataset(train_path)

# 查看图片读取效果
import matplotlib.pyplot as plt

o_img, l_img = traindata[np.random.randint(0, 2000)]
plt.subplot(1, 2, 1)
plt.imshow(o_img.permute(1, 2, 0))
plt.subplot(1, 2, 2)
plt.imshow(l_img.permute(1, 2, 0))
print("原始图片张量的形状:", o_img.shape)
print("标签图片张量的形状:", l_img.shape)

# 配置模型超参数
# 模型保存的路径
model_path = 'models/'
# 推荐使用gpu进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 学习率
lr = 1e-4
# 学习率衰减
weight_decay = 1e-3
# 批大小
bs = 8
# 训练轮次
epochs = 50


# 数据 Backbone部分使用了MixVisionTransformer(mit-b1)
# 分割头一开始用的是DeeplabV3+， 但是效果不太好，后面尝试了不同的分割头发现Manet的效果是最好的。
import segmentation_models_pytorch as smp

model = smp.MAnet(
    encoder_name="efficientnet-b1",
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
)

# 训练前准备
from torch.utils.data import DataLoader

# 加载模型到gpu
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
trainloader = DataLoader(traindata, batch_size=bs, shuffle=True, num_workers=0)
print("训练集的大小:", len(traindata))

# 经典dice loss损失函数，借鉴别人的
def dice_loss(logits, target):
    smooth = 1.
    prob = torch.sigmoid(logits)
    batch = prob.size(0)
    prob = prob.view(batch, 1, -1)
    target = target.view(batch, 1, -1)
    intersection = torch.sum(prob * target, dim=2)
    denominator = torch.sum(prob, dim=2) + torch.sum(target, dim=2)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    dice = torch.mean(dice)
    dice_loss = 1. - dice
    return dice_loss


loss_last = 100000
best_model_name = 'x'
# 记录loss变化
for epoch in range(1, epochs + 1):
    for step, (inputs, labels) in tqdm(enumerate(trainloader), desc=f"Epoch {epoch}/{epochs}",
                                       ascii=True, total=len(trainloader)):
        # 原始图片和标签
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        loss = dice_loss(out, labels)
        # 后向
        optim.zero_grad()
        # 梯度反向传播
        loss.backward()
        optim.step()
    # 损失小于上一轮则添加
    if loss < loss_last:
        loss_last = loss
        torch.save(model.state_dict(), model_path + 'model_epoch{}_loss{}.pth'.format(epoch, loss))
        best_model_name = model_path + 'model_epoch{}_loss{}.pth'.format(epoch, loss)
    print(f"\nEpoch: {epoch}/{epochs},DiceLoss:{loss}")
print("训练集的大小:", len(traindata))
