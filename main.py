import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset
from PIL import Image

# 取消提示
import warnings
warnings.filterwarnings("ignore")

# 使用GPU或者CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 定义data类型，处理多标签
class MyDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = {}
            for line in f.readlines():
                filename, label_list = line.strip().split()
                labels = [int(x) for x in label_list.split(',')]
                # 如果标签不足7个，则用0填充
                while len(labels) < 7:
                    labels.append(0)
                self.labels[filename] = labels
        self.filenames = list(self.labels.keys())
        self.transform = transform
        self.label_names = ['有角', '有斑点', '有犬牙', '有蹄','有羽毛','有爪子','有长尾']
        self.class_to_idx = {label_name: idx for idx, label_name in enumerate(self.label_names)}

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.filenames[index])
        img = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[self.filenames[index]])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)


# 训练集图像预处理：旋转、水平翻转、垂直翻转、裁剪----（图像增强）、转 Tensor、归一化
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(45),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])# 均值，标准差
                                     ])

# 测试集图像预处理：缩放、转 Tensor、归一化
test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])# 均值，标准差
                                    ])
# 处理训练集
train_data = MyDataset(img_dir = 'D:\\多标签动物特征识别\\data\\train\\train-1',
                       label_file = 'D:\\多标签动物特征识别\\data\\train\\train-labels.txt',
                       transform = train_transforms)

# 处理测试集
test_data = MyDataset(img_dir = 'D:\\多标签动物特征识别\\data\\test\\test-1',
                       label_file = 'D:\\多标签动物特征识别\\data\\test\\test-labels.txt',
                       transform = test_transforms)

# 数据加载器
from torch.utils.data import DataLoader
BATCH_SIZE = 32                                  # GPU为 GTX 1650Ti,跑不了太高

# 训练集的数据加载器
train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True,          # 随机打乱顺序，增加多样性
                          num_workers=0          # 单个线程
                         )

# 测试集的数据加载器
test_loader = DataLoader(test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=False,          # 测试集不用，为了提高准确性
                         num_workers=0
                        )

print('训练集图像数量', len(train_data))
print('标签维度', train_data[0][1].shape[0])
print('测试集图像数量', len(test_data))
print('标签维度', train_data[0][1].shape[0])

class_name = train_data.label_names              # 调出类别名称
n_class = len(class_name)                        # 输出个数

# 映射关系：类别 到 索引号
print(train_data.class_to_idx)
# 映射关系：索引号 到 类别
idx_to_labels = {y:x for x,y in train_data.class_to_idx.items()}

# 保存为本地的 npy 文件----方便读取数据，更快响应
np.save('D:\\多标签动物特征识别\\npy文件\\idx_to_labels.npy', idx_to_labels)            # 保存 索引号 到 类别
np.save('D:\\多标签动物特征识别\\npy文件\\labels_to_idx.npy', train_data.class_to_idx)  # 保存 类别 到 索引号

# 导入迁移学习模型
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

model = models.resnet50(pretrained=True)                           # 载入预训练模型，使用它的权重作为初始值
model.fc = nn.Linear(model.fc.in_features, n_class)
optimizer = optim.Adam(model.parameters())                          # 优化器选择Adam
model = model.to(device)

# 二元交叉熵损失函数
criterion = nn.BCEWithLogitsLoss()

# 训练轮次 Epoch
EPOCHS = 50

# 学习率降低策略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 初始化最高准确率
best_accuracy = 0

# 遍历每个 EPOCH,进行训练
for epoch in tqdm(range(EPOCHS)):
    model.train()
    for images, labels in train_loader:         # 获取训练集的一个 batch，包含数据和标注
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)                 # 前向预测，获得当前 batch 的预测结果
        labels = labels.float()
        loss = criterion(outputs, labels)       # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数

        optimizer.zero_grad()
        loss.backward()                         # 损失函数对神经网络权重反向传播求梯度
        optimizer.step()                        # 优化更新神经网络权重


# 测试集进行测试，返回准确率
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    true_positives = torch.zeros(len(class_name)).to(device)
    false_negatives = torch.zeros(len(class_name)).to(device)
    false_positives = torch.zeros(len(class_name)).to(device)

    for images, labels in tqdm(test_loader):  # 获取测试集的一个 batch，包含数据和标注
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.float()
        outputs = model(images)  # 前向预测，获得当前 batch 的预测置信度

        preds = (torch.sigmoid(outputs) > 0.5).float()  # 设置阈值为0.5，将置信度转换为二进制标签
        total += labels.size(0)

        true_positives += ((preds == 1) & (labels == 1)).sum(axis=0).float()
        false_negatives += torch.clamp((labels.sum(axis=0) > 0).float().sum(axis=0) - true_positives, min=0)
        false_positives += torch.clamp((preds.sum(axis=0) > 0).float().sum(axis=0) - true_positives, min=0)
        correct += ((preds == 1) & (labels == 1)).sum(axis=1).sum().item() + ((preds == 0) & (labels == 0)).sum(
            axis=1).sum().item()

        accuracy = 100 * correct / (total * len(class_name))                                          # 计算准确率
        recall = 100 * true_positives / (true_positives + false_negatives)                            # 计算召回率
        precision = 100 * true_positives / (true_positives + false_positives)                         # 计算精确率
        f1_score = 0.02 * precision * recall / (precision + recall)                                   # 计算F1值

        print('测试集上的准确率为 {:.3f} %'.format(accuracy))
        print('测试集上的召回率为 {:.3f} %'.format(recall.mean().item()))
        print('测试集上的精确率为 {:.3f} %'.format(precision.mean().item()))
        print('测试集上的F1值为 {:.3f}'.format(f1_score.mean().item()))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    lr_scheduler.step()

# 保存模型
torch.save(model, "C:\\Users\\26028\\.cache\\torch\\hub\\checkpoints/AnimalCheck-9.pth")
