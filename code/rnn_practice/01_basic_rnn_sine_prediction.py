"""
01_basic_rnn_sine_prediction.py

任务：
使用过去连续 20 个正弦函数值，
预测下一个正弦函数值。

这是 RNN 入门的第一段代码，重点学习：
1. 序列数据是什么；
2. 如何制作滑动窗口；
3. RNN 输入张量的形状；
4. nn.RNN 的基本使用；
5. output 和 h_n 的含义；
6. 使用最后一个时间步进行预测；
7. RNN 的训练、测试和保存。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# ============================================================
# 1. 设置随机种子
# ============================================================

# 固定随机种子后，多次运行代码时，
# 模型初始化和数据打乱结果会更加接近。
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 2. 设置训练参数
# ============================================================

# 每个输入序列包含多少个连续数值
sequence_length = 20

# RNN 隐藏状态的特征数量
hidden_size = 32

# RNN 堆叠层数
num_layers = 1

# 每个 batch 包含多少个样本
batch_size = 32

# 完整训练集重复训练多少次
epochs = 60

# 学习率
learning_rate = 0.005

# 训练集比例
train_ratio = 0.8


# ============================================================
# 3. 选择训练设备
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前使用的设备：", device)


# ============================================================
# 4. 创建保存结果的文件夹
# ============================================================

output_dir = Path("E:/Python/ChemML/RNN_practice/outputs")
output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 5. 生成人工时间序列
# ============================================================

# 创建 1200 个从 0 到 40 的连续时间点
time_values = np.linspace(
    0,
    40,
    1200,
    dtype=np.float32,
)

# 计算每个时间点对应的正弦函数值
series_values = np.sin(time_values).astype(
    np.float32
)

print("\n原始时间序列形状：")
print(series_values.shape)

print("\n前 10 个时间序列数值：")
print(series_values[:10])


# ============================================================
# 6. 使用滑动窗口制作 RNN 数据
# ============================================================

def create_sequences(
    series: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将一条完整时间序列转换为多个监督学习样本。

    例如原始序列为：

    [0.1, 0.2, 0.3, 0.4, 0.5]

    如果 sequence_length = 3，那么：

    第一个样本：
    X = [0.1, 0.2, 0.3]
    y = 0.4

    第二个样本：
    X = [0.2, 0.3, 0.4]
    y = 0.5
    """

    X_sequences = []
    y_targets = []

    # 每次向后移动一个位置
    for start_index in range(
        len(series) - sequence_length
    ):
        end_index = start_index + sequence_length

        # 连续 sequence_length 个数值作为输入
        sequence = series[
            start_index:end_index
        ]

        # 序列后面的一个数值作为预测目标
        target = series[end_index]

        X_sequences.append(sequence)
        y_targets.append(target)

    return (
        np.array(X_sequences),
        np.array(y_targets),
    )


X_numpy, y_numpy = create_sequences(
    series=series_values,
    sequence_length=sequence_length,
)

print("\n滑动窗口处理后的 X 形状：")
print(X_numpy.shape)

print("\n滑动窗口处理后的 y 形状：")
print(y_numpy.shape)

print("\n第一个输入序列：")
print(X_numpy[0])

print("\n第一个序列对应的预测目标：")
print(y_numpy[0])


# ============================================================
# 7. 转换为 PyTorch 张量
# ============================================================

X_tensor = torch.tensor(
    X_numpy,
    dtype=torch.float32,
)

y_tensor = torch.tensor(
    y_numpy,
    dtype=torch.float32,
)


# ------------------------------------------------------------
# RNN 的输入需要三个维度：
#
# [样本数, 序列长度, 每个时间步的特征数]
#
# 当前 X_tensor 的形状是：
#
# [样本数, 序列长度]
#
# 每个时间步只有一个数值，因此特征数为 1。
# 使用 unsqueeze(-1) 在最后增加一个维度。
# ------------------------------------------------------------

X_tensor = X_tensor.unsqueeze(-1)

# 将 y 从 [样本数] 变成 [样本数, 1]
y_tensor = y_tensor.unsqueeze(-1)

print("\n转换后的 X 张量形状：")
print(X_tensor.shape)

print("\n转换后的 y 张量形状：")
print(y_tensor.shape)


# ============================================================
# 8. 按照时间顺序划分训练集和测试集
# ============================================================

total_samples = len(X_tensor)

train_size = int(
    total_samples * train_ratio
)

# 前 80% 作为训练集
X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]

# 后 20% 作为测试集
X_test = X_tensor[train_size:]
y_test = y_tensor[train_size:]

print("\n训练集 X 形状：")
print(X_train.shape)

print("训练集 y 形状：")
print(y_train.shape)

print("\n测试集 X 形状：")
print(X_test.shape)

print("测试集 y 形状：")
print(y_test.shape)


# ============================================================
# 9. 创建 Dataset 和 DataLoader
# ============================================================

train_dataset = TensorDataset(
    X_train,
    y_train,
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)


# ============================================================
# 10. 定义最基础的 RNN 模型
# ============================================================

class BasicRNN(nn.Module):
    """
    最基础的 RNN 回归模型。

    模型结构：

    输入序列
        ↓
    RNN 层
        ↓
    取最后一个时间步的输出
        ↓
    全连接层
        ↓
    预测下一个数值
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ----------------------------------------------------
        # input_size：
        # 每个时间步输入多少个特征。
        #
        # 本代码每个时间点只有一个正弦值，
        # 所以 input_size = 1。
        #
        # hidden_size：
        # RNN 隐藏状态中包含多少个特征。
        #
        # num_layers：
        # 堆叠多少层 RNN。
        #
        # batch_first=True：
        # 输入形状使用：
        # [batch, sequence, feature]
        # ----------------------------------------------------

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
        )

        # 将 RNN 最后一个时间步的隐藏特征
        # 转换成一个预测数值
        self.output_layer = nn.Linear(
            in_features=hidden_size,
            out_features=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x 的形状：

        [batch_size, sequence_length, input_size]
        """

        # 将完整序列输入 RNN
        rnn_output, h_n = self.rnn(x)

        # rnn_output 形状：
        # [batch_size, sequence_length, hidden_size]
        #
        # 它保存每个时间步的输出。

        # h_n 形状：
        # [num_layers, batch_size, hidden_size]
        #
        # 它保存每一层 RNN 最后一个时间步的隐藏状态。

        # 取每个样本最后一个时间步的输出
        last_time_step = rnn_output[:, -1, :]

        # last_time_step 形状：
        # [batch_size, hidden_size]

        # 使用全连接层得到最终预测
        prediction = self.output_layer(
            last_time_step
        )

        # prediction 形状：
        # [batch_size, 1]

        return prediction


# ============================================================
# 11. 创建模型
# ============================================================

model = BasicRNN(
    input_size=1,
    hidden_size=hidden_size,
    num_layers=num_layers,
)

model = model.to(device)

print("\n模型结构：")
print(model)


# ============================================================
# 12. 查看一个 batch 经过 RNN 后的形状变化
# ============================================================

example_X_batch, example_y_batch = next(
    iter(train_loader)
)

example_X_batch = example_X_batch.to(device)

with torch.no_grad():
    example_rnn_output, example_h_n = model.rnn(
        example_X_batch
    )

    example_prediction = model(
        example_X_batch
    )

print("\n一个 batch 的输入形状：")
print(example_X_batch.shape)

print("\n一个 batch 的标签形状：")
print(example_y_batch.shape)

print("\nRNN 的全部时间步输出形状：")
print(example_rnn_output.shape)

print("\nRNN 最终隐藏状态 h_n 的形状：")
print(example_h_n.shape)

print("\n模型最终预测形状：")
print(example_prediction.shape)


# ============================================================
# 13. 定义损失函数和优化器
# ============================================================

# 回归任务使用均方误差损失
loss_function = nn.MSELoss()

# Adam 优化器负责更新模型参数
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# ============================================================
# 14. 训练模型
# ============================================================

train_loss_history = []
test_loss_history = []

print("\n开始训练：")

for epoch in range(1, epochs + 1):

    # --------------------------------------------------------
    # 训练模式
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:

        # 将数据移动到 CPU 或 GPU
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # 清空上一次反向传播留下的梯度
        optimizer.zero_grad()

        # 前向传播
        predictions = model(batch_X)

        # 计算损失
        loss = loss_function(
            predictions,
            batch_y,
        )

        # 反向传播，计算梯度
        loss.backward()

        # 根据梯度更新参数
        optimizer.step()

        # 累加当前 batch 的损失
        total_train_loss += (
            loss.item() * batch_X.size(0)
        )

    average_train_loss = (
        total_train_loss / len(train_dataset)
    )

    train_loss_history.append(
        average_train_loss
    )

    # --------------------------------------------------------
    # 测试模式
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        test_predictions = model(
            X_test.to(device)
        )

        test_loss = loss_function(
            test_predictions,
            y_test.to(device),
        )

    test_loss_value = test_loss.item()

    test_loss_history.append(
        test_loss_value
    )

    # 每 10 个 epoch 输出一次结果
    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"训练损失：{average_train_loss:.6f} | "
            f"测试损失：{test_loss_value:.6f}"
        )


# ============================================================
# 15. 最终测试
# ============================================================

model.eval()

with torch.no_grad():

    test_predictions = model(
        X_test.to(device)
    )

# 将预测结果移动回 CPU
test_predictions = test_predictions.cpu()

# 计算评价指标
mse = torch.mean(
    (test_predictions - y_test) ** 2
).item()

rmse = mse ** 0.5

mae = torch.mean(
    torch.abs(test_predictions - y_test)
).item()

print("\n最终测试结果：")
print(f"MSE：  {mse:.6f}")
print(f"RMSE： {rmse:.6f}")
print(f"MAE：  {mae:.6f}")


# ============================================================
# 16. 查看部分真实值和预测值
# ============================================================

print("\n前 10 个测试样本的预测结果：")

for index in range(10):

    true_value = y_test[index].item()

    predicted_value = (
        test_predictions[index].item()
    )

    print(
        f"样本 {index + 1:2d} | "
        f"真实值：{true_value: .6f} | "
        f"预测值：{predicted_value: .6f}"
    )


# ============================================================
# 17. 绘制训练损失曲线
# ============================================================

loss_figure_path = (
    output_dir / "01_basic_rnn_loss_curve.png"
)

plt.figure(figsize=(10, 5))

plt.plot(
    train_loss_history,
    label="Train loss",
)

plt.plot(
    test_loss_history,
    label="Test loss",
)

plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Basic RNN Loss Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    loss_figure_path,
    dpi=300,
)

plt.show()
plt.close()


# ============================================================
# 18. 绘制预测结果
# ============================================================

prediction_figure_path = (
    output_dir / "01_basic_rnn_predictions.png"
)

true_values = y_test.squeeze(-1).numpy()

predicted_values = (
    test_predictions.squeeze(-1).numpy()
)

# 只画前 200 个测试样本，方便观察
show_count = min(
    200,
    len(true_values),
)

plt.figure(figsize=(12, 5))

plt.plot(
    true_values[:show_count],
    label="True values",
)

plt.plot(
    predicted_values[:show_count],
    label="Predicted values",
)

plt.xlabel("Test sample")
plt.ylabel("Value")
plt.title("Basic RNN Sine Wave Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    prediction_figure_path,
    dpi=300,
)

plt.show()
plt.close()


# ============================================================
# 19. 保存模型参数
# ============================================================

model_path = (
    output_dir / "01_basic_rnn_model.pth"
)

torch.save(
    model.state_dict(),
    model_path,
)

print("\n文件保存完成：")
print("模型：", model_path)
print("损失曲线：", loss_figure_path)
print("预测结果图：", prediction_figure_path)