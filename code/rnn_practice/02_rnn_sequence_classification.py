"""
02_rnn_sequence_classification.py

任务：把一整段波形输入 RNN，判断它是正弦波、方波还是锯齿波。

重点知识：
1. 序列到类别（many-to-one classification）
2. h_n 最终隐藏状态
3. logits、CrossEntropyLoss、argmax
4. 训练集、验证集、测试集
5. 早停、混淆矩阵、模型保存与加载
"""

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 随机种子与参数
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

sequence_length = 60
samples_per_class = 400
num_classes = 3
class_names = ["Sine", "Square", "Sawtooth"]

hidden_size = 48
num_layers = 1
batch_size = 64
epochs = 60
learning_rate = 0.003
early_stopping_patience = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)

# 输出目录：E:\Python\ChemML\RNN_practice\outputs
# 使用脚本相对路径，项目整体搬家后依然有效
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 生成三类人工波形
# ============================================================

def create_wave_sample(class_index: int, length: int) -> np.ndarray:
    """生成一条带随机振幅、相位和噪声的波形。"""

    t = np.linspace(
        0.0,
        2.0 * np.pi,
        length,
        endpoint=False,
        dtype=np.float32,
    )

    amplitude = rng.uniform(0.8, 1.2)
    phase = rng.uniform(-0.5, 0.5)
    shifted_t = t + phase

    if class_index == 0:
        wave = np.sin(shifted_t)
    elif class_index == 1:
        wave = np.sign(np.sin(shifted_t))
    elif class_index == 2:
        wave = 2.0 * ((shifted_t / (2.0 * np.pi)) % 1.0) - 1.0
    else:
        raise ValueError(f"未知类别编号：{class_index}")

    noise = rng.normal(0.0, 0.05, size=length)
    return (amplitude * wave + noise).astype(np.float32)


X_list: list[np.ndarray] = []
y_list: list[int] = []

for class_index in range(num_classes):
    for _ in range(samples_per_class):
        X_list.append(create_wave_sample(class_index, sequence_length))
        y_list.append(class_index)

X_numpy = np.array(X_list, dtype=np.float32)
y_numpy = np.array(y_list, dtype=np.int64)

print("\n原始 X 形状：", X_numpy.shape)  # [样本数, 序列长度]
print("原始 y 形状：", y_numpy.shape)  # [样本数]


# ============================================================
# 3. 打乱并划分训练、验证、测试集
# ============================================================

indices = rng.permutation(len(X_numpy))
X_numpy = X_numpy[indices]
y_numpy = y_numpy[indices]

train_end = int(len(X_numpy) * 0.70)
validation_end = int(len(X_numpy) * 0.85)

X_train_numpy = X_numpy[:train_end]
y_train_numpy = y_numpy[:train_end]

X_validation_numpy = X_numpy[train_end:validation_end]
y_validation_numpy = y_numpy[train_end:validation_end]

X_test_numpy = X_numpy[validation_end:]
y_test_numpy = y_numpy[validation_end:]

print("\n训练样本数：", len(X_train_numpy))
print("验证样本数：", len(X_validation_numpy))
print("测试样本数：", len(X_test_numpy))


# ============================================================
# 4. 标准化：只能用训练集计算参数
# ============================================================

train_mean = float(X_train_numpy.mean())
train_std = float(X_train_numpy.std())

if train_std < 1e-8:
    raise ValueError("训练集标准差太小，无法标准化。")

X_train_numpy = (X_train_numpy - train_mean) / train_std
X_validation_numpy = (X_validation_numpy - train_mean) / train_std
X_test_numpy = (X_test_numpy - train_mean) / train_std


# ============================================================
# 5. 转换为张量与 DataLoader
# ============================================================

def make_dataset(X_array: np.ndarray, y_array: np.ndarray) -> TensorDataset:
    # RNN 输入形状：[batch, sequence, feature]
    # 每个时间步只有一个数值，因此 feature=1。
    X_tensor = torch.tensor(X_array, dtype=torch.float32).unsqueeze(-1)

    # CrossEntropyLoss 的类别标签必须是整数 long 类型。
    y_tensor = torch.tensor(y_array, dtype=torch.long)

    return TensorDataset(X_tensor, y_tensor)


train_dataset = make_dataset(X_train_numpy, y_train_numpy)
validation_dataset = make_dataset(X_validation_numpy, y_validation_numpy)
test_dataset = make_dataset(X_test_numpy, y_test_numpy)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# 6. RNN 分类模型
# ============================================================

class RNNClassifier(nn.Module):
    """输入整段序列，输出三个类别的 logits。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rnn_output: [batch, sequence, hidden_size]
        # h_n: [num_layers, batch, hidden_size]
        rnn_output, h_n = self.rnn(x)

        # 取最后一层 RNN 的最终隐藏状态。
        last_hidden_state = h_n[-1]  # [batch, hidden_size]

        # logits 是未经 Softmax 的原始类别分数。
        logits = self.classifier(last_hidden_state)  # [batch, num_classes]
        return logits


model = RNNClassifier(
    input_size=1,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
).to(device)

print("\n模型结构：")
print(model)


# ============================================================
# 7. 检查一个 batch 的形状
# ============================================================

example_X, example_y = next(iter(train_loader))
example_X = example_X.to(device)

with torch.no_grad():
    example_rnn_output, example_h_n = model.rnn(example_X)
    example_logits = model(example_X)

print("\n一个 batch 的 X：", example_X.shape)
print("一个 batch 的 y：", example_y.shape)
print("全部时间步输出：", example_rnn_output.shape)
print("最终隐藏状态 h_n：", example_h_n.shape)
print("最终 logits：", example_logits.shape)


# ============================================================
# 8. 损失函数、优化器与评估函数
# ============================================================

# CrossEntropyLoss 内部已经包含适合分类的 LogSoftmax，
# 训练时不要先对 logits 手动使用 Softmax。
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def evaluate_model(
    evaluated_model: nn.Module,
    data_loader: DataLoader,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    evaluated_model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = evaluated_model(batch_X)
            loss = loss_function(logits, batch_y)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item() * batch_X.size(0)
            correct_count += (predictions == batch_y).sum().item()
            sample_count += batch_X.size(0)

            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

    return (
        total_loss / sample_count,
        correct_count / sample_count,
        torch.cat(all_predictions).numpy(),
        torch.cat(all_targets).numpy(),
    )


# ============================================================
# 9. 训练、验证与早停
# ============================================================

train_loss_history: list[float] = []
validation_loss_history: list[float] = []
train_accuracy_history: list[float] = []
validation_accuracy_history: list[float] = []

best_validation_loss = float("inf")
best_model_state: dict[str, torch.Tensor] | None = None
epochs_without_improvement = 0

print("\n开始训练：")

for epoch in range(1, epochs + 1):
    model.train()

    total_train_loss = 0.0
    train_correct_count = 0
    train_sample_count = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = loss_function(logits, batch_y)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_train_loss += loss.item() * batch_X.size(0)
        train_correct_count += (predictions == batch_y).sum().item()
        train_sample_count += batch_X.size(0)

    train_loss = total_train_loss / train_sample_count
    train_accuracy = train_correct_count / train_sample_count

    validation_loss, validation_accuracy, _, _ = evaluate_model(
        model,
        validation_loader,
    )

    train_loss_history.append(train_loss)
    validation_loss_history.append(validation_loss)
    train_accuracy_history.append(train_accuracy)
    validation_accuracy_history.append(validation_accuracy)

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_model_state = deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"训练损失：{train_loss:.4f} | "
            f"验证损失：{validation_loss:.4f} | "
            f"训练准确率：{train_accuracy:.2%} | "
            f"验证准确率：{validation_accuracy:.2%}"
        )

    if epochs_without_improvement >= early_stopping_patience:
        print(
            f"\n验证集连续 {early_stopping_patience} 轮没有改善，提前停止。"
        )
        break

if best_model_state is None:
    raise RuntimeError("没有获得最佳模型参数。")

model.load_state_dict(best_model_state)


# ============================================================
# 10. 测试与混淆矩阵
# ============================================================

test_loss, test_accuracy, test_predictions, test_targets = evaluate_model(
    model,
    test_loader,
)

print("\n最终测试结果：")
print(f"测试损失：{test_loss:.4f}")
print(f"测试准确率：{test_accuracy:.2%}")

print("\n前 15 个测试样本：")
for index in range(min(15, len(test_targets))):
    print(
        f"样本 {index + 1:2d} | "
        f"真实：{class_names[test_targets[index]]:9s} | "
        f"预测：{class_names[test_predictions[index]]:9s}"
    )

confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
for true_class, predicted_class in zip(test_targets, test_predictions):
    confusion_matrix[true_class, predicted_class] += 1

print("\n混淆矩阵（行=真实类别，列=预测类别）：")
print(confusion_matrix)


# ============================================================
# 11. 绘图
# ============================================================

loss_figure_path = output_dir / "02_rnn_classification_loss.png"
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train loss")
plt.plot(validation_loss_history, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("RNN Classification Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_figure_path, dpi=300)
plt.show()
plt.close()

accuracy_figure_path = output_dir / "02_rnn_classification_accuracy.png"
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy_history, label="Train accuracy")
plt.plot(validation_accuracy_history, label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("RNN Classification Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(accuracy_figure_path, dpi=300)
plt.show()
plt.close()

confusion_figure_path = output_dir / "02_rnn_classification_confusion_matrix.png"
plt.figure(figsize=(7, 6))
plt.imshow(confusion_matrix)
plt.xticks(range(num_classes), class_names)
plt.yticks(range(num_classes), class_names)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion Matrix")

for row in range(num_classes):
    for column in range(num_classes):
        plt.text(
            column,
            row,
            str(confusion_matrix[row, column]),
            ha="center",
            va="center",
        )

plt.colorbar()
plt.tight_layout()
plt.savefig(confusion_figure_path, dpi=300)
plt.show()
plt.close()


# ============================================================
# 12. 保存、加载并预测新样本
# ============================================================

model_path = output_dir / "02_rnn_wave_classifier.pth"
torch.save(model.state_dict(), model_path)

loaded_model = RNNClassifier(
    input_size=1,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
).to(device)

# 如果旧版 PyTorch 不支持 weights_only，请删除该参数。
loaded_state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True,
)
loaded_model.load_state_dict(loaded_state_dict)
loaded_model.eval()

new_sample_numpy = create_wave_sample(class_index=2, length=sequence_length)
new_sample_numpy = (new_sample_numpy - train_mean) / train_std

# [sequence] -> [1, sequence, 1]
new_sample_tensor = (
    torch.tensor(new_sample_numpy, dtype=torch.float32)
    .unsqueeze(0)
    .unsqueeze(-1)
    .to(device)
)

with torch.no_grad():
    new_logits = loaded_model(new_sample_tensor)
    new_probabilities = torch.softmax(new_logits, dim=1)
    new_prediction = torch.argmax(new_logits, dim=1).item()

print("\n新样本预测：")
print("真实类别：Sawtooth")
print("预测类别：", class_names[new_prediction])

for class_name, probability in zip(
    class_names,
    new_probabilities[0].cpu().numpy(),
):
    print(f"{class_name:9s}: {probability:.2%}")

print("\n保存完成：")
print("模型：", model_path)
print("损失曲线：", loss_figure_path)
print("准确率曲线：", accuracy_figure_path)
print("混淆矩阵：", confusion_figure_path)