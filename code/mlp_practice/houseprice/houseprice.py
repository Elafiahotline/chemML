from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. 固定随机种子
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. 创建文件夹和文件路径
# ============================================================

data_dir = Path(__file__).resolve().parent
output_dir = Path(__file__).resolve().parent

data_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

data_path = data_dir / "house_price_data.csv"
history_path = output_dir / "training_history.csv"
prediction_path = output_dir / "test_predictions.csv"
model_path = output_dir / "house_price_mlp.pth"


# ============================================================
# 3. 生成模拟房屋数据
# ============================================================

number_of_houses = 500

# 面积：50～200平方米，连续变量
area = np.random.uniform(
    low=50,
    high=200,
    size=number_of_houses,
)

# 卧室数量：1～5间，整数变量
bedrooms = np.random.randint(
    low=1,
    high=6,
    size=number_of_houses,
)

# 房龄：0～30年，连续变量
age = np.random.uniform(
    low=0,
    high=30,
    size=number_of_houses,
)

# 随机噪声：平均值为0，标准差为2
noise = np.random.normal(
    loc=0,
    scale=2,
    size=number_of_houses,
)

# 模拟房价，单位为万元
#
# area ** 2 是一个非线性项：
# 面积对房价的影响并非永远保持完全相同
price = (
    0.08 * area
    + 2.5 * bedrooms
    - 0.18 * age
    + 0.00035 * area**2
    + noise
)


# ============================================================
# 4. 保存模拟数据为CSV
# ============================================================

house_data = pd.DataFrame(
    {
        "area": area,
        "bedrooms": bedrooms,
        "age": age,
        "price": price,
    }
)

house_data.to_csv(
    data_path,
    index=False,
)

print("模拟数据保存位置：")
print(data_path.resolve())


# ============================================================
# 5. 从CSV读取数据
# ============================================================

data = pd.read_csv(data_path)

print("\n数据前5行：")
print(data.head())

print("\n整个DataFrame的形状：")
print(data.shape)


# ============================================================
# 6. 构造输入特征X和目标y
# ============================================================

feature_columns = [
    "area",
    "bedrooms",
    "age",
]

target_column = "price"

# X有3列：
# 面积、卧室数、房龄
X = data[feature_columns].to_numpy(
    dtype=np.float32,
)

# y只有1列：房价
y = data[target_column].to_numpy(
    dtype=np.float32,
).reshape(-1, 1)

print("\nX的形状：", X.shape)
print("y的形状：", y.shape)

print("\n第一个房屋的特征：", X[0])
print("第一个房屋的价格：", y[0])


# ============================================================
# 7. 划分训练集和测试集
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\n训练集X形状：", X_train.shape)
print("测试集X形状：", X_test.shape)


# ============================================================
# 8. 标准化输入特征
# ============================================================

scaler = StandardScaler()

# 只能用训练集计算每一列的均值和标准差
X_train_scaled = scaler.fit_transform(X_train)

# 测试集使用训练集计算出的标准化规则
X_test_scaled = scaler.transform(X_test)

print("\n训练集各列原始均值：")
print(scaler.mean_)

print("训练集各列原始标准差：")
print(scaler.scale_)


# ============================================================
# 9. 转换成PyTorch Tensor
# ============================================================

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32,
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32,
)


# ============================================================
# 10. 创建Dataset和DataLoader
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)


# ============================================================
# 11. 定义全连接神经网络
# ============================================================

class HousePriceMLP(nn.Module):

    def __init__(self, input_dim: int):
        # 初始化父类nn.Module
        super().__init__()

        # 创建并保存网络结构
        self.network = nn.Sequential(
            # 每套房屋3个输入特征
            # 转换成32个隐藏特征
            nn.Linear(input_dim, 32),
            nn.ReLU(),

            # 32个隐藏特征转换成16个
            nn.Linear(32, 16),
            nn.ReLU(),

            # 16个隐藏特征转换成1个房价
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# X.shape[1]是特征数量，这里等于3
model = HousePriceMLP(
    input_dim=X.shape[1],
)

print("\n模型结构：")
print(model)


# ============================================================
# 12. 查看一个batch的形状变化
# ============================================================

example_X_batch, example_y_batch = next(
    iter(train_loader)
)

print("\n一个batch的输入形状：")
print(example_X_batch.shape)

print("一个batch的标签形状：")
print(example_y_batch.shape)

with torch.no_grad():
    example_predictions = model(example_X_batch)

print("经过模型后的预测形状：")
print(example_predictions.shape)


# ============================================================
# 13. 定义损失函数和优化器
# ============================================================

# 回归任务使用MSE作为训练损失
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# ============================================================
# 14. 训练模型
# ============================================================

epochs = 300

history = []

for epoch in range(epochs):

    model.train()

    # 当前epoch中所有样本的损失总和
    total_loss = 0.0

    # 当前epoch中处理的样本数量
    total_samples = 0

    for X_batch, y_batch in train_loader:

        # 清空上一批留下的梯度
        optimizer.zero_grad()

        # 前向传播
        predictions = model(X_batch)

        # 计算当前batch的平均MSE
        loss = criterion(
            predictions,
            y_batch,
        )

        # 反向传播，计算梯度
        loss.backward()

        # 根据梯度更新模型参数
        optimizer.step()

        current_batch_size = X_batch.size(0)

        # loss.item()是当前batch的平均损失
        # 乘以batch_size，得到这一批样本的损失总和
        total_loss += loss.item() * current_batch_size

        total_samples += current_batch_size

    # 当前epoch中所有样本的平均损失
    average_loss = total_loss / total_samples

    history.append(
        {
            "epoch": epoch + 1,
            "loss": average_loss,
        }
    )

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1:3d}，"
            f"平均Loss：{average_loss:.4f}"
        )


# ============================================================
# 15. 保存训练历史
# ============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    history_path,
    index=False,
)


# ============================================================
# 16. 在测试集上预测
# ============================================================

model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor)

y_pred = test_predictions.numpy()
y_true = y_test_tensor.numpy()


# ============================================================
# 17. 计算评价指标
# ============================================================

mae = mean_absolute_error(
    y_true,
    y_pred,
)

mse = mean_squared_error(
    y_true,
    y_pred,
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_true,
    y_pred,
)

print("\n测试集评价结果：")
print(f"MAE： {mae:.4f} 万元")
print(f"MSE： {mse:.4f}")
print(f"RMSE：{rmse:.4f} 万元")
print(f"R²：  {r2:.4f}")


# ============================================================
# 18. 保存测试集预测结果
# ============================================================

prediction_result = pd.DataFrame(
    {
        "true_price": y_true.reshape(-1),
        "predicted_price": y_pred.reshape(-1),
        "absolute_error": np.abs(
            y_true.reshape(-1)
            - y_pred.reshape(-1)
        ),
    }
)

prediction_result.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# 19. 保存模型参数
# ============================================================

torch.save(
    model.state_dict(),
    model_path,
)


# ============================================================
# 20. 预测一套新房屋
# ============================================================

new_house = np.array(
    [
        [
            120,  # 面积
            3,    # 卧室数
            5,    # 房龄
        ]
    ],
    dtype=np.float32,
)

# 新数据必须使用训练集的scaler进行标准化
new_house_scaled = scaler.transform(new_house)

new_house_tensor = torch.tensor(
    new_house_scaled,
    dtype=torch.float32,
)

model.eval()

with torch.no_grad():
    predicted_price = model(new_house_tensor)

print(
    "\n新房屋预测价格："
    f"{predicted_price.item():.2f} 万元"
)


# ============================================================
# 21. 输出生成文件的位置
# ============================================================

print("\n生成的文件：")
print("模拟数据：", data_path.resolve())
print("训练历史：", history_path.resolve())
print("测试预测：", prediction_path.resolve())
print("模型参数：", model_path.resolve())