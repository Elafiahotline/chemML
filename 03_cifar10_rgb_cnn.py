"""
03_cifar10_rgb_cnn.py

用途
----
使用卷积神经网络 CNN 对 CIFAR-10 彩色图像进行分类。

CIFAR-10 包含 10 个类别：
airplane、automobile、bird、cat、deer、
dog、frog、horse、ship、truck。

本项目学习内容
--------------
1. RGB 三通道图像；
2. 数据增强；
3. 训练集、验证集和测试集；
4. CNN、BatchNorm、Dropout；
5. CrossEntropyLoss；
6. AdamW 优化器；
7. 学习率调度；
8. Early Stopping；
9. 保存最佳模型；
10. 混淆矩阵和错误分析。
"""

from pathlib import Path
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# 1. 设置随机种子
# ============================================================

RANDOM_SEED = 42


def set_random_seed(seed: int) -> None:
    """
    固定随机种子，使多次运行的结果尽量接近。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seed(RANDOM_SEED)


# ============================================================
# 2. 文件路径
# ============================================================

# 项目目录：以脚本所在位置为基准
project_dir = Path(__file__).resolve().parent / "cifar10_project"

data_dir = project_dir / "data"
output_dir = project_dir / "outputs"
figure_dir = output_dir / "figures"

data_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)

history_path = output_dir / "training_history.csv"
prediction_path = output_dir / "test_predictions.csv"
class_accuracy_path = output_dir / "class_accuracy.csv"
model_path = output_dir / "cifar10_cnn_best.pth"

loss_figure_path = figure_dir / "loss_curve.png"
accuracy_figure_path = figure_dir / "accuracy_curve.png"
confusion_matrix_path = figure_dir / "confusion_matrix.png"
mistake_figure_path = figure_dir / "misclassified_examples.png"


# ============================================================
# 3. 训练参数
# ============================================================

batch_size = 128

# 第一次测试代码时可以先改成2
# 确认代码正常后再改成30
epochs = 30

learning_rate = 0.001
weight_decay = 1e-4

# 从原训练集中取5000张图片作为验证集
validation_size = 5000

# 验证集连续7轮没有改善，就停止训练
early_stopping_patience = 7


# ============================================================
# 4. 选择CPU或GPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)

if device.type == "cuda":
    print("GPU名称：", torch.cuda.get_device_name(0))


# ============================================================
# 5. 定义图像预处理
# ============================================================

# 三个0.5分别对应R、G、B三个通道
image_mean = (
    0.5,
    0.5,
    0.5,
)

image_std = (
    0.5,
    0.5,
    0.5,
)


# 训练集预处理：
# 包含随机裁剪和随机水平翻转
train_transform = transforms.Compose(
    [
        # 先在图片四周增加4个像素，
        # 再随机裁剪回32×32
        transforms.RandomCrop(
            size=32,
            padding=4,
        ),

        # 以50%的概率水平翻转图片
        transforms.RandomHorizontalFlip(
            p=0.5,
        ),

        # PIL图片转换为Tensor
        transforms.ToTensor(),

        # 分别标准化R、G、B通道
        transforms.Normalize(
            mean=image_mean,
            std=image_std,
        ),
    ]
)


# 验证集和测试集不能进行随机数据增强
evaluation_transform = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(
            mean=image_mean,
            std=image_std,
        ),
    ]
)


# ============================================================
# 6. 下载并读取CIFAR-10
# ============================================================

# 这个数据集用于训练，因此包含随机数据增强
full_train_augmented_dataset = datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=True,
    transform=train_transform,
)


# 这个数据集与上面的图片完全相同，
# 但不包含随机数据增强，用于构造验证集
full_train_evaluation_dataset = datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=False,
    transform=evaluation_transform,
)


# 官方测试集
test_dataset = datasets.CIFAR10(
    root=data_dir,
    train=False,
    download=True,
    transform=evaluation_transform,
)


# CIFAR-10类别名称
class_names = full_train_augmented_dataset.classes

print("\nCIFAR-10类别：")

for class_index, class_name in enumerate(class_names):
    print(
        class_index,
        class_name,
    )


# ============================================================
# 7. 划分训练集和验证集
# ============================================================

# 使用固定随机种子打乱所有索引
split_generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

all_indices = torch.randperm(
    len(full_train_augmented_dataset),
    generator=split_generator,
).tolist()


# 前5000个索引作为验证集
validation_indices = all_indices[
    :validation_size
]

# 剩余索引作为训练集
training_indices = all_indices[
    validation_size:
]


# 训练集使用带数据增强的数据集
train_dataset = Subset(
    full_train_augmented_dataset,
    training_indices,
)


# 验证集使用不带数据增强的数据集
validation_dataset = Subset(
    full_train_evaluation_dataset,
    validation_indices,
)


print("\n数据集大小：")
print("训练集：", len(train_dataset))
print("验证集：", len(validation_dataset))
print("测试集：", len(test_dataset))


# ============================================================
# 8. 创建DataLoader
# ============================================================

# CUDA下使用pin_memory通常有利于CPU到GPU的数据传输
use_pin_memory = device.type == "cuda"


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,

    # Windows初学阶段使用0最稳定
    num_workers=0,

    pin_memory=use_pin_memory,
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=use_pin_memory,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=use_pin_memory,
)


# ============================================================
# 9. 查看一个batch的数据形状
# ============================================================

example_images, example_labels = next(
    iter(train_loader)
)

print("\n一个batch的图片形状：")
print(example_images.shape)

print("一个batch的标签形状：")
print(example_labels.shape)

print("第一张图片的标签编号：")
print(example_labels[0].item())

print("第一张图片的类别：")
print(
    class_names[
        example_labels[0].item()
    ]
)


# ============================================================
# 10. 定义CNN模型
# ============================================================

class CIFAR10CNN(nn.Module):

    def __init__(
        self,
        number_of_classes: int,
    ):
        super().__init__()

        # ----------------------------------------------------
        # 卷积特征提取部分
        # ----------------------------------------------------

        self.features = nn.Sequential(

            # 输入：
            # [batch, 3, 32, 32]
            #
            # 输出：
            # [batch, 32, 32, 32]
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 输出尺寸：
            # [batch, 32, 16, 16]
            nn.MaxPool2d(
                kernel_size=2,
            ),

            nn.Dropout2d(
                p=0.10,
            ),


            # 第二个卷积块
            #
            # 输入：
            # [batch, 32, 16, 16]
            #
            # 输出：
            # [batch, 64, 16, 16]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 输出尺寸：
            # [batch, 64, 8, 8]
            nn.MaxPool2d(
                kernel_size=2,
            ),

            nn.Dropout2d(
                p=0.15,
            ),


            # 第三个卷积块
            #
            # 输入：
            # [batch, 64, 8, 8]
            #
            # 输出：
            # [batch, 128, 8, 8]
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 输出尺寸：
            # [batch, 128, 4, 4]
            nn.MaxPool2d(
                kernel_size=2,
            ),


            # 把每张128通道的4×4特征图
            # 平均压缩成128通道的1×1特征
            #
            # 输出：
            # [batch, 128, 1, 1]
            nn.AdaptiveAvgPool2d(
                output_size=(1, 1),
            ),
        )


        # ----------------------------------------------------
        # 分类部分
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            # [batch, 128, 1, 1]
            # 转换成
            # [batch, 128]
            nn.Flatten(),

            nn.Dropout(
                p=0.30,
            ),

            # 输出10个类别的分数
            nn.Linear(
                in_features=128,
                out_features=number_of_classes,
            ),
        )


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)
        x = self.classifier(x)

        return x


model = CIFAR10CNN(
    number_of_classes=len(class_names),
).to(device)


print("\n模型结构：")
print(model)


number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

number_of_trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n模型总参数量：")
print(number_of_parameters)

print("可训练参数量：")
print(number_of_trainable_parameters)


# ============================================================
# 11. 查看模型输入输出形状
# ============================================================

model.eval()

with torch.no_grad():

    example_logits = model(
        example_images.to(device)
    )


print("\n一个batch经过模型后的输出形状：")
print(example_logits.shape)

print(
    "含义：",
    f"{batch_size}张图片，",
    f"每张图片输出{len(class_names)}个类别分数。",
)


# ============================================================
# 12. 损失函数、优化器和学习率调度器
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)


# 当验证集Loss连续若干轮没有改善时，
# 自动把学习率乘以0.5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)


# ============================================================
# 13. 定义单轮训练函数
# ============================================================

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练模型一轮。

    返回
    ----
    average_loss:
        当前训练轮的平均损失。

    accuracy:
        当前训练轮的分类准确率。
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # 清空上一批数据留下的梯度
        optimizer.zero_grad(
            set_to_none=True,
        )

        # 前向传播
        logits = model(images)

        # 计算损失
        loss = criterion(
            logits,
            labels,
        )

        # 反向传播
        loss.backward()

        # 更新模型参数
        optimizer.step()

        current_batch_size = images.size(0)

        total_loss += (
            loss.item()
            * current_batch_size
        )

        predicted_labels = logits.argmax(
            dim=1
        )

        total_correct += (
            predicted_labels
            == labels
        ).sum().item()

        total_samples += current_batch_size


    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return average_loss, accuracy


# ============================================================
# 14. 定义验证或测试函数
# ============================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    在验证集或测试集上评价模型。
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            current_batch_size = images.size(0)

            total_loss += (
                loss.item()
                * current_batch_size
            )

            predicted_labels = logits.argmax(
                dim=1
            )

            total_correct += (
                predicted_labels
                == labels
            ).sum().item()

            total_samples += current_batch_size


    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return average_loss, accuracy


# ============================================================
# 15. 训练模型
# ============================================================

history = []

best_validation_loss = float("inf")
best_model_state = None

epochs_without_improvement = 0


for epoch in range(epochs):

    train_loss, train_accuracy = train_one_epoch(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )


    validation_loss, validation_accuracy = evaluate_model(
        model=model,
        data_loader=validation_loader,
        criterion=criterion,
        device=device,
    )


    # 根据验证集Loss调整学习率
    scheduler.step(
        validation_loss
    )


    current_learning_rate = (
        optimizer.param_groups[0]["lr"]
    )


    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "learning_rate": current_learning_rate,
        }
    )


    print(
        f"Epoch {epoch + 1:3d}/{epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_accuracy * 100:.2f}%, "
        f"Val Loss: {validation_loss:.4f}, "
        f"Val Acc: {validation_accuracy * 100:.2f}%, "
        f"LR: {current_learning_rate:.6f}"
    )


    # --------------------------------------------------------
    # 保存验证集Loss最低的模型
    # --------------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        print("保存新的最佳模型。")

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # Early Stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= early_stopping_patience
    ):

        print(
            "\n验证集连续"
            f"{early_stopping_patience}轮没有改善，"
            "提前停止训练。"
        )

        break


# ============================================================
# 16. 恢复并保存最佳模型
# ============================================================

if best_model_state is None:
    raise RuntimeError(
        "没有保存到最佳模型，请检查训练过程。"
    )


model.load_state_dict(
    best_model_state
)


torch.save(
    best_model_state,
    model_path,
)


print("\n最佳验证集Loss：")
print(f"{best_validation_loss:.4f}")

print("最佳模型保存位置：")
print(model_path.resolve())


# ============================================================
# 17. 保存训练历史
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    history_path,
    index=False,
)


# ============================================================
# 18. 绘制Loss曲线
# ============================================================

figure, axis = plt.subplots(
    figsize=(8, 5)
)

axis.plot(
    history_df["epoch"],
    history_df["train_loss"],
    label="Train Loss",
)

axis.plot(
    history_df["epoch"],
    history_df["validation_loss"],
    label="Validation Loss",
)

axis.set_xlabel("Epoch")
axis.set_ylabel("Loss")
axis.set_title("CIFAR-10 Training and Validation Loss")
axis.legend()
axis.grid(alpha=0.3)

figure.tight_layout()

figure.savefig(
    loss_figure_path,
    dpi=300,
)

plt.close(figure)


# ============================================================
# 19. 绘制准确率曲线
# ============================================================

figure, axis = plt.subplots(
    figsize=(8, 5)
)

axis.plot(
    history_df["epoch"],
    history_df["train_accuracy"] * 100,
    label="Train Accuracy",
)

axis.plot(
    history_df["epoch"],
    history_df["validation_accuracy"] * 100,
    label="Validation Accuracy",
)

axis.set_xlabel("Epoch")
axis.set_ylabel("Accuracy (%)")
axis.set_title("CIFAR-10 Training and Validation Accuracy")
axis.legend()
axis.grid(alpha=0.3)

figure.tight_layout()

figure.savefig(
    accuracy_figure_path,
    dpi=300,
)

plt.close(figure)


# ============================================================
# 20. 在测试集上评价
# ============================================================

test_loss, test_accuracy = evaluate_model(
    model=model,
    data_loader=test_loader,
    criterion=criterion,
    device=device,
)


print("\n测试集结果：")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ============================================================
# 21. 收集测试集预测结果
# ============================================================

all_true_labels = []
all_predicted_labels = []

# 保存一部分错误预测样本
misclassified_examples = []


model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images_device = images.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images_device
        )

        predicted_labels = logits.argmax(
            dim=1
        ).cpu()


        all_true_labels.extend(
            labels.numpy().tolist()
        )

        all_predicted_labels.extend(
            predicted_labels.numpy().tolist()
        )


        # 找出当前batch中预测错误的位置
        incorrect_positions = (
            predicted_labels
            != labels
        ).nonzero(
            as_tuple=False
        ).reshape(-1)


        for position in incorrect_positions:

            if len(misclassified_examples) >= 12:
                break

            position_index = position.item()

            misclassified_examples.append(
                (
                    images[position_index].clone(),
                    labels[position_index].item(),
                    predicted_labels[position_index].item(),
                )
            )


all_true_labels = np.array(
    all_true_labels
)

all_predicted_labels = np.array(
    all_predicted_labels
)


# ============================================================
# 22. 保存测试集预测结果
# ============================================================

prediction_result = pd.DataFrame(
    {
        "true_label_id": all_true_labels,
        "true_label": [
            class_names[label]
            for label in all_true_labels
        ],
        "predicted_label_id": all_predicted_labels,
        "predicted_label": [
            class_names[label]
            for label in all_predicted_labels
        ],
        "correct": (
            all_true_labels
            == all_predicted_labels
        ),
    }
)


prediction_result.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# 23. 计算混淆矩阵
# ============================================================

confusion = confusion_matrix(
    all_true_labels,
    all_predicted_labels,
    labels=list(
        range(
            len(class_names)
        )
    ),
)


figure, axis = plt.subplots(
    figsize=(10, 8)
)

matrix_image = axis.imshow(
    confusion
)

figure.colorbar(
    matrix_image,
    ax=axis,
)

axis.set_xticks(
    range(
        len(class_names)
    )
)

axis.set_yticks(
    range(
        len(class_names)
    )
)

axis.set_xticklabels(
    class_names,
    rotation=45,
    ha="right",
)

axis.set_yticklabels(
    class_names
)

axis.set_xlabel("Predicted Label")
axis.set_ylabel("True Label")
axis.set_title("CIFAR-10 Confusion Matrix")


threshold = confusion.max() / 2

for row_index in range(
    len(class_names)
):

    for column_index in range(
        len(class_names)
    ):

        value = confusion[
            row_index,
            column_index,
        ]

        axis.text(
            column_index,
            row_index,
            str(value),
            ha="center",
            va="center",
            color=(
                "white"
                if value > threshold
                else "black"
            ),
        )


figure.tight_layout()

figure.savefig(
    confusion_matrix_path,
    dpi=300,
)

plt.close(figure)


# ============================================================
# 24. 计算每个类别的准确率
# ============================================================

class_correct = np.diag(
    confusion
)

class_total = confusion.sum(
    axis=1
)

class_accuracy = (
    class_correct
    / class_total
)


class_accuracy_result = pd.DataFrame(
    {
        "class_id": range(
            len(class_names)
        ),
        "class_name": class_names,
        "correct_count": class_correct,
        "total_count": class_total,
        "accuracy": class_accuracy,
    }
)


class_accuracy_result.to_csv(
    class_accuracy_path,
    index=False,
)


print("\n每个类别的准确率：")

for class_name, accuracy in zip(
    class_names,
    class_accuracy,
):

    print(
        f"{class_name:10s}: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# 25. 绘制错误预测图片
# ============================================================

if misclassified_examples:

    figure, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(12, 9),
    )

    axes = axes.reshape(-1)

    for axis, example in zip(
        axes,
        misclassified_examples,
    ):

        image_tensor, true_id, predicted_id = example

        # 反标准化：
        # normalized = (original - 0.5) / 0.5
        #
        # 因此：
        # original = normalized * 0.5 + 0.5
        image_tensor = (
            image_tensor
            * 0.5
            + 0.5
        )

        # [C, H, W]
        # 转换成
        # [H, W, C]
        image_array = image_tensor.permute(
            1,
            2,
            0,
        ).numpy()

        image_array = np.clip(
            image_array,
            0.0,
            1.0,
        )

        axis.imshow(
            image_array
        )

        axis.set_title(
            f"True: {class_names[true_id]}\n"
            f"Pred: {class_names[predicted_id]}"
        )

        axis.axis("off")


    # 如果错误样本不足12个，关闭多余坐标轴
    for axis in axes[
        len(misclassified_examples):
    ]:
        axis.axis("off")


    figure.tight_layout()

    figure.savefig(
        mistake_figure_path,
        dpi=300,
    )

    plt.close(figure)


# ============================================================
# 26. 输出文件位置
# ============================================================

print("\n生成的文件：")
print("训练历史：", history_path.resolve())
print("测试预测：", prediction_path.resolve())
print("类别准确率：", class_accuracy_path.resolve())
print("最佳模型：", model_path.resolve())
print("Loss曲线：", loss_figure_path.resolve())
print("准确率曲线：", accuracy_figure_path.resolve())
print("混淆矩阵：", confusion_matrix_path.resolve())
print("错误预测图：", mistake_figure_path.resolve())