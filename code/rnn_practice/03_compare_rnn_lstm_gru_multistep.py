"""
03_compare_rnn_lstm_gru_multistep.py

任务：使用三路传感器过去 20 个时间步的数据，预测未来 5 个目标值。
代码会分别训练并比较 RNN、LSTM、GRU。

重点知识：
1. 多特征序列输入与多步回归输出
2. RNN、LSTM、GRU 的统一接口
3. LSTM 的 h_n 与 c_n
4. 时间顺序划分数据，避免未来信息泄漏
5. 早停、学习率调度、梯度裁剪
6. MAE、RMSE、模型比较、保存、加载与预测
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
torch.set_num_threads(1)
rng = np.random.default_rng(RANDOM_SEED)

sequence_length = 20
prediction_horizon = 5
input_size = 3
hidden_size = 16
num_layers = 1

batch_size = 128
epochs = 15
learning_rate = 0.003
early_stopping_patience = 4
gradient_clip_max_norm = 1.0

model_types = ["RNN", "LSTM", "GRU"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：", device)

# 输出目录：E:\Python\ChemML\RNN_practice\outputs
# 使用脚本相对路径，项目整体搬家后依然有效
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 生成人工多变量时间序列
# ============================================================

time_values = np.linspace(0.0, 35.0, 600, dtype=np.float32)

# 需要预测的目标信号。
clean_target = (
    np.sin(time_values)
    + 0.35 * np.sin(0.22 * time_values + 0.6)
    + 0.004 * time_values
).astype(np.float32)

# 三路传感器分别提供不同但相关的信息。
sensor_1 = (
    clean_target + rng.normal(0.0, 0.06, size=len(time_values))
).astype(np.float32)

sensor_2 = (
    np.cos(time_values) + rng.normal(0.0, 0.06, size=len(time_values))
).astype(np.float32)

sensor_3 = (
    0.35 * np.sin(0.22 * time_values + 0.6)
    + 0.004 * time_values
    + rng.normal(0.0, 0.04, size=len(time_values))
).astype(np.float32)

# [时间点, 特征数] = [600, 3]
all_features = np.stack([sensor_1, sensor_2, sensor_3], axis=1).astype(
    np.float32
)

print("\n完整特征形状：", all_features.shape)
print("完整目标形状：", clean_target.shape)


# ============================================================
# 3. 滑动窗口：过去 30 步预测未来 5 步
# ============================================================

def create_multistep_sequences(
    features: np.ndarray,
    target: np.ndarray,
    input_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    X: [样本数, input_length, input_size]
    y: [样本数, horizon]
    """

    X_sequences: list[np.ndarray] = []
    y_sequences: list[np.ndarray] = []

    sample_count = len(features) - input_length - horizon + 1

    for start_index in range(sample_count):
        input_end = start_index + input_length
        target_end = input_end + horizon

        X_sequences.append(features[start_index:input_end])
        y_sequences.append(target[input_end:target_end])

    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_sequences, dtype=np.float32),
    )


X_numpy, y_numpy = create_multistep_sequences(
    features=all_features,
    target=clean_target,
    input_length=sequence_length,
    horizon=prediction_horizon,
)

print("\n滑动窗口后的 X：", X_numpy.shape)
print("滑动窗口后的 y：", y_numpy.shape)


# ============================================================
# 4. 按时间顺序划分训练、验证、测试集
# ============================================================

# 时间序列不能先随机打乱再划分，否则未来信息可能进入训练集。
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
# 5. 标准化：参数只能来自训练集
# ============================================================

# 对三个输入特征分别计算均值和标准差。
X_mean = X_train_numpy.mean(axis=(0, 1), keepdims=True)
X_std = X_train_numpy.std(axis=(0, 1), keepdims=True)
X_std = np.where(X_std < 1e-8, 1.0, X_std)

# 目标只有一种物理量，因此使用一个均值和标准差。
y_mean = float(y_train_numpy.mean())
y_std = float(y_train_numpy.std())

if y_std < 1e-8:
    raise ValueError("目标标准差太小，无法标准化。")

X_train_scaled = (X_train_numpy - X_mean) / X_std
X_validation_scaled = (X_validation_numpy - X_mean) / X_std
X_test_scaled = (X_test_numpy - X_mean) / X_std

y_train_scaled = (y_train_numpy - y_mean) / y_std
y_validation_scaled = (y_validation_numpy - y_mean) / y_std
y_test_scaled = (y_test_numpy - y_mean) / y_std

print("\n输入特征均值：", X_mean.reshape(-1))
print("输入特征标准差：", X_std.reshape(-1))
print("目标均值：", y_mean)
print("目标标准差：", y_std)


# ============================================================
# 6. Dataset 与 DataLoader
# ============================================================

def make_dataset(X_array: np.ndarray, y_array: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X_array, dtype=torch.float32),
        torch.tensor(y_array, dtype=torch.float32),
    )


train_dataset = make_dataset(X_train_scaled, y_train_scaled)
validation_dataset = make_dataset(X_validation_scaled, y_validation_scaled)
test_dataset = make_dataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# 7. 统一的 RNN、LSTM、GRU 模型
# ============================================================

class RecurrentForecaster(nn.Module):
    """model_type 可选 RNN、LSTM、GRU。"""

    def __init__(
        self,
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()

        model_type = model_type.upper()
        if model_type not in {"RNN", "LSTM", "GRU"}:
            raise ValueError("model_type 必须是 RNN、LSTM 或 GRU。")

        self.model_type = model_type

        common_arguments = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
        }

        if model_type == "RNN":
            self.recurrent_layer = nn.RNN(
                **common_arguments,
                nonlinearity="tanh",
            )
        elif model_type == "LSTM":
            self.recurrent_layer = nn.LSTM(**common_arguments)
        else:
            self.recurrent_layer = nn.GRU(**common_arguments)

        # 把最终隐藏状态转换成未来 5 个预测值。
        self.output_network = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent_output, hidden = self.recurrent_layer(x)

        if self.model_type == "LSTM":
            # LSTM 返回 (h_n, c_n)：
            # h_n 是最终隐藏状态，c_n 是最终细胞状态。
            h_n, c_n = hidden
        else:
            # 普通 RNN 和 GRU 只返回 h_n。
            h_n = hidden

        last_hidden_state = h_n[-1]  # [batch, hidden_size]
        predictions = self.output_network(last_hidden_state)
        return predictions  # [batch, prediction_horizon]


# ============================================================
# 8. 评估函数
# ============================================================

loss_function = nn.MSELoss()


def evaluate_scaled_loss(model: nn.Module, data_loader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)

            total_loss += loss.item() * batch_X.size(0)
            total_samples += batch_X.size(0)

    return total_loss / total_samples


def predict_original_scale(
    model: nn.Module,
    data_loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            predictions_scaled = model(batch_X.to(device))
            all_predictions.append(predictions_scaled.cpu())
            all_targets.append(batch_y)

    predictions_scaled = torch.cat(all_predictions).numpy()
    targets_scaled = torch.cat(all_targets).numpy()

    predictions_original = predictions_scaled * y_std + y_mean
    targets_original = targets_scaled * y_std + y_mean

    return predictions_original, targets_original


# ============================================================
# 9. 训练一个指定类型的循环网络
# ============================================================

def train_one_model(
    model_type: str,
) -> tuple[nn.Module, list[float], list[float]]:
    print("\n" + "=" * 60)
    print(f"开始训练 {model_type}")
    print("=" * 60)

    model = RecurrentForecaster(
        model_type=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=prediction_horizon,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # 验证损失长期不下降时，将学习率减半。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    train_history: list[float] = []
    validation_history: list[float] = []

    best_validation_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_sample_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()

            # 防止循环网络训练时出现梯度爆炸。
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_max_norm,
            )

            optimizer.step()

            total_train_loss += loss.item() * batch_X.size(0)
            train_sample_count += batch_X.size(0)

        train_loss = total_train_loss / train_sample_count
        validation_loss = evaluate_scaled_loss(model, validation_loader)

        train_history.append(train_loss)
        validation_history.append(validation_loss)
        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_learning_rate = optimizer.param_groups[0]["lr"]

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"训练损失：{train_loss:.6f} | "
                f"验证损失：{validation_loss:.6f} | "
                f"学习率：{current_learning_rate:.6f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"{model_type} 提前停止：验证集连续 "
                f"{early_stopping_patience} 轮没有改善。"
            )
            break

    if best_model_state is None:
        raise RuntimeError(f"{model_type} 没有获得最佳参数。")

    model.load_state_dict(best_model_state)
    return model, train_history, validation_history


# ============================================================
# 10. 检查三个模型的输入输出形状
# ============================================================

example_X, example_y = next(iter(train_loader))
example_X = example_X.to(device)

print("\n一个 batch 的输入：", example_X.shape)
print("一个 batch 的目标：", example_y.shape)

for model_type in model_types:
    temporary_model = RecurrentForecaster(
        model_type=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=prediction_horizon,
    ).to(device)

    with torch.no_grad():
        temporary_output = temporary_model(example_X)

    print(f"{model_type} 输出：", temporary_output.shape)


# ============================================================
# 11. 训练并比较 RNN、LSTM、GRU
# ============================================================

trained_models: dict[str, nn.Module] = {}
training_histories: dict[str, dict[str, list[float]]] = {}
comparison_results: dict[str, dict[str, object]] = {}

for model_type in model_types:
    trained_model, train_history, validation_history = train_one_model(
        model_type
    )

    predictions, targets = predict_original_scale(trained_model, test_loader)
    errors = predictions - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    trained_models[model_type] = trained_model
    training_histories[model_type] = {
        "train": train_history,
        "validation": validation_history,
    }
    comparison_results[model_type] = {
        "mae": mae,
        "rmse": rmse,
        "predictions": predictions,
        "targets": targets,
    }

    model_path = output_dir / f"03_{model_type.lower()}_multistep.pth"
    torch.save(trained_model.state_dict(), model_path)

    print(f"\n{model_type} 测试结果：")
    print(f"MAE： {mae:.6f}")
    print(f"RMSE：{rmse:.6f}")
    print("模型已保存：", model_path)


# ============================================================
# 12. 模型排名
# ============================================================

ranking = sorted(
    comparison_results.items(),
    key=lambda item: float(item[1]["rmse"]),
)

print("\n模型 RMSE 排名：")
for rank, (model_type, result) in enumerate(ranking, start=1):
    print(
        f"{rank}. {model_type:4s} | "
        f"MAE：{float(result['mae']):.6f} | "
        f"RMSE：{float(result['rmse']):.6f}"
    )

best_model_type = ranking[0][0]
print("\n本次运行测试集表现最好的是：", best_model_type)


# ============================================================
# 13. 绘制验证损失和预测结果
# ============================================================

loss_figure_path = output_dir / "03_rnn_lstm_gru_validation_loss.png"
plt.figure(figsize=(11, 6))

for model_type in model_types:
    plt.plot(
        training_histories[model_type]["validation"],
        label=f"{model_type} validation",
    )

plt.xlabel("Epoch")
plt.ylabel("Scaled MSE loss")
plt.title("RNN vs LSTM vs GRU Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(loss_figure_path, dpi=300)
plt.show()
plt.close()

prediction_figure_path = output_dir / "03_rnn_lstm_gru_predictions.png"
show_count = 200

reference_targets = np.asarray(
    comparison_results[model_types[0]]["targets"]
).reshape(-1)

plt.figure(figsize=(12, 6))
plt.plot(reference_targets[:show_count], label="True values")

for model_type in model_types:
    flat_predictions = np.asarray(
        comparison_results[model_type]["predictions"]
    ).reshape(-1)
    plt.plot(flat_predictions[:show_count], label=model_type)

plt.xlabel("Flattened future step")
plt.ylabel("Target value")
plt.title("RNN vs LSTM vs GRU Predictions")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(prediction_figure_path, dpi=300)
plt.show()
plt.close()


# ============================================================
# 14. 重新加载最佳模型并预测未来 5 步
# ============================================================

best_model_path = output_dir / f"03_{best_model_type.lower()}_multistep.pth"
loaded_best_model = RecurrentForecaster(
    model_type=best_model_type,
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=prediction_horizon,
).to(device)

# 如果旧版 PyTorch 不支持 weights_only，请删除该参数。
loaded_state_dict = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True,
)
loaded_best_model.load_state_dict(loaded_state_dict)
loaded_best_model.eval()

# X_test_scaled[0] 原形状：[sequence_length, input_size]
# 增加 batch 维度后：[1, sequence_length, input_size]
new_X_tensor = (
    torch.tensor(X_test_scaled[0], dtype=torch.float32)
    .unsqueeze(0)
    .to(device)
)

with torch.no_grad():
    future_scaled = loaded_best_model(new_X_tensor)

future_prediction = future_scaled.cpu().numpy()[0] * y_std + y_mean
true_future = y_test_numpy[0]

print("\n最佳模型的新样本预测：")
print("模型类型：", best_model_type)

for future_step in range(prediction_horizon):
    print(
        f"未来第 {future_step + 1} 步 | "
        f"真实值：{true_future[future_step]: .6f} | "
        f"预测值：{future_prediction[future_step]: .6f}"
    )

print("\n图片保存完成：")
print("验证损失比较：", loss_figure_path)
print("预测结果比较：", prediction_figure_path)