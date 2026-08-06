"""
01_mnist_simple_cnn.py

用途：
使用最简单的卷积神经网络 CNN，
识别 MNIST 数据集中的手写数字 0～9。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

# 脚本所在目录：数据、模型和输出都基于这个目录
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. 设置随机种子
# ============================================================

torch.manual_seed(42)


# ============================================================
# 2. 设置训练参数
# ============================================================

batch_size = 64
epochs = 5
learning_rate = 0.001


# 如果有 GPU 就使用 GPU，否则使用 CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)


# ============================================================
# 3. 定义图片预处理
# ============================================================

transform = transforms.ToTensor()


# ============================================================
# 4. 下载并读取 MNIST 数据集
# ============================================================

train_dataset = datasets.MNIST(
    root=SCRIPT_DIR / "data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root=SCRIPT_DIR / "data",
    train=False,
    download=True,
    transform=transform,
)


# ============================================================
# 5. 创建 DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)


# ============================================================
# 6. 定义最简单的 CNN
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 卷积层：
        # 输入1个通道，输出8个特征图
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 最大池化层：
        # 图片长和宽都缩小一半
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        # 全连接输出层：
        # 8 × 14 × 14 个特征
        # 转换成10个类别分数
        self.fc = nn.Linear(
            8 * 14 * 14,
            10,
        )

    def forward(self, x):

        # 输入形状：
        # [batch_size, 1, 28, 28]

        x = self.conv(x)

        # 卷积后：
        # [batch_size, 8, 28, 28]

        x = torch.relu(x)

        x = self.pool(x)

        # 池化后：
        # [batch_size, 8, 14, 14]

        x = torch.flatten(
            x,
            start_dim=1,
        )

        # 展平后：
        # [batch_size, 8 × 14 × 14]
        # 也就是 [batch_size, 1568]

        x = self.fc(x)

        # 输出：
        # [batch_size, 10]
        # 每张图片对应10个类别分数

        return x


# ============================================================
# 7. 创建模型
# ============================================================

model = SimpleCNN().to(device)

print("\n模型结构：")
print(model)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\n模型参数数量：", total_parameters)


# ============================================================
# 8. 定义损失函数和优化器
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# ============================================================
# 9. 训练模型
# ============================================================

for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    for images, labels in train_loader:

        # 将数据移动到 CPU 或 GPU
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一个batch留下的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算分类损失
        loss = criterion(
            outputs,
            labels,
        )

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        current_batch_size = images.size(0)

        total_loss += (
            loss.item() * current_batch_size
        )

        total_samples += current_batch_size

        # 找出分数最高的类别
        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

    average_loss = total_loss / total_samples

    train_accuracy = (
        correct_predictions
        / total_samples
    )

    print(
        f"Epoch {epoch + 1:2d}/{epochs}，"
        f"Loss：{average_loss:.4f}，"
        f"训练准确率：{train_accuracy:.2%}"
    )


# ============================================================
# 10. 在测试集上评估
# ============================================================

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)


test_accuracy = test_correct / test_total

print(
    f"\n测试集准确率：{test_accuracy:.2%}"
)

model_dir = SCRIPT_DIR / "models"

model_dir.mkdir(
    exist_ok=True
)


model_path = model_dir / "mnist_cnn.pth"


torch.save(
    model.state_dict(),
    model_path
)


print(
    f"模型已保存到：{model_path}"
)

# ============================================================
# 11. 显示部分预测结果
# ============================================================

images, labels = next(iter(test_loader))

sample_images = images[:10].to(device)
sample_labels = labels[:10]

model.eval()

with torch.no_grad():
    sample_outputs = model(sample_images)

sample_predictions = (
    sample_outputs
    .argmax(dim=1)
    .cpu()
)

plt.figure(figsize=(12, 4))

for index in range(10):

    plt.subplot(2, 5, index + 1)

    plt.imshow(
        images[index].squeeze(),
        cmap="gray",
    )

    plt.title(
        f"真实：{sample_labels[index].item()}\n"
        f"预测：{sample_predictions[index].item()}"
    )

    plt.axis("off")

plt.tight_layout()


output_dir = SCRIPT_DIR / "outputs"

output_dir.mkdir(
    exist_ok=True
)


plt.tight_layout()


plt.savefig(
    output_dir / "prediction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()