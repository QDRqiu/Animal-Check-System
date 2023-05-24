import torch
import torch.nn.functional as F
import numpy as np

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 设置matplotlib中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# # 导入pillow中文字体
from PIL import ImageFont
# 导入中文字体，指定字号
font = ImageFont.truetype('SimHei.ttf', 32)

# 载入类别
idx_to_labels = np.load(r"D:\多标签动物特征识别\npy文件\idx_to_labels.npy", allow_pickle=True).item()

# 导入训练好的模型
model = torch.load(r"C:\Users\26028\.cache\torch\hub\checkpoints\AnimalCheck-8.pth")
model = model.eval().to(device)

# 预处理
from torchvision import transforms
# 测试集图像预处理：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                     ])

# 载入一张测试图像
from PIL import Image,ImageDraw
img_path = r"C:\Users\26028\Desktop\测试图片\train\fd2b02b4c614454a.jpg"
img_pil = Image.open(img_path)

# 预处理
input_img = test_transform(img_pil)  # 预处理
input_img = input_img.unsqueeze(0).to(device)

# 执行前向预测，得到所有类别的 logit 预测分数
pred_logits = model(input_img)

# 对 logit 分数做 sigmoid 运算
pred_sigmoid = F.sigmoid(pred_logits)

# 置信度最大的前n个结果
n = 7                                                   # 我只设置了 7 个结果
top_n = torch.topk(pred_sigmoid, n)                     # 取置信度的 n 个结果
pred_ids = top_n[1].cpu().detach().numpy().squeeze()    # 解析出类别
confs = top_n[0].cpu().detach().numpy().squeeze()       # 解析出置信度
print(pred_ids)

# 图像分类结果写在原图上
draw = ImageDraw.Draw(img_pil)
for i in range(n):
    class_name = idx_to_labels[pred_ids[i]]  # 获取类别名称
    confidence = confs[i]   # 获取置信度
    text = '{:<15} {:>.4f}'.format(class_name, confidence)
    print(text)
    # 文字坐标，中文字符串，字体，红色
    draw.text((50, 50 + 50 * i), text, font = font,fill=(255, 0, 0, 1))
Image._show(img_pil)