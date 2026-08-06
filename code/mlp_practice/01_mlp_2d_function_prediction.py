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

# 1.随机种子
version = 4

np.random.seed(42)
torch.manual_seed(42)

#2.文件路径

data_dir = Path(__file__).resolve().parent / "2d"
output_dir = Path(__file__).resolve().parent / "2d"

data_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

data_path = data_dir / f"z_data_v{version}.csv"
history_path = output_dir / f"training_history_v{version}.csv"
prediction_path = output_dir / f"test_predictions_v{version}.csv"
model_path = output_dir / f"z_mlp_v{version}.pth"

#3.生成模拟数据

if version == 1:

    x = torch.rand(10000,1)*6-3
    y = torch.rand(10000,1)*6-3

elif version in [2,3,4]:

    x = torch.rand(10000,1)*20-10
    y = torch.rand(10000,1)*20-10

if version == 1 or version == 2:

    z = torch.sin(x) + torch.cos(y)

elif version == 3:

    z = torch.sin(x*y) + 0.1*x**2 + torch.cos(y)

elif version == 4:

    z = torch.sin(x*y) + 0.1*x**2 + torch.cos(y)

    noise = torch.randn_like(z)*0.2

    z = z + noise

#4.保存生成数据

z_data = pd.DataFrame(
    {
        "x":x.numpy().reshape(-1),
        "y":y.numpy().reshape(-1),
        "z":z.numpy().reshape(-1),
    }
)

z_data.to_csv(
    data_path,
    index=False,
)

print("模拟数据保存位置：")
print(data_path.resolve())

#5.从csv读取数据

data = pd.read_csv(data_path)

print("\n数据前5行：")
print(data.head())

print("\n整个DataFrame的形状：")
print(data.shape)

#6.构造特征x和y

feature_columns = [
    "x",
    "y",
]

target_column = "z"

X = data[feature_columns].to_numpy(
    dtype=np.float32,
)

y = data[target_column].to_numpy(
    dtype=np.float32,
).reshape(-1, 1)

print("\nX:", X.shape)
print("y:", y.shape)

print("\n第一个x:", X[0])
print("第一个y:", y[0])

#7.划分训练集和测试集

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\n训练集X形状：", X_train.shape)
print("测试集X形状：", X_test.shape)

#8.标准化输入特征

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

print("\n训练集各列原始均值：")
print(X_scaler.mean_)

print("训练集各列原始标准差：")
print(X_scaler.scale_)

#9.转换成pyTorch Tensor

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32,
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)

y_test_tensor = torch.tensor(
    y_test_scaled,
    dtype=torch.float32,
)

#10.创建Dataset和DataLoader

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)

#11.定义全连接神经网络

class ZMLP(nn.Module):

    def __init__(self,input_dim:int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,1),
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.network(x)

model = ZMLP(
    input_dim=X.shape[1],
)

print("\n模型结构:")
print(model)

#12.查看一个batch的形状变化

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

#13.定义损失函数和优化器

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

#14.训练模型

epochs = 300

history = []

for epoch in range(epochs):

    model.train()

    total_loss = 0.0
    total_sample = 0

    for X_batch,y_batch in train_loader:

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            y_batch,
        )

        loss.backward()

        optimizer.step()

        current_batch_size = X_batch.size(0)

        total_loss += loss.item() * current_batch_size

        total_sample += current_batch_size

    average_loss = total_loss / total_sample

    history.append(
        {
            "epoch":epoch + 1,
            "loss":average_loss
        }
    )

    if(epoch + 1) % 50 == 0:
        print(
            f"Epoch{epoch + 1:3d},"
            f"平均Loss:{average_loss:.4f}"
        )

#15.保存训练历史

history_df = pd.DataFrame(history)

history_df.to_csv(
    history_path,
    index=False,
)

#16.预测集上预测

model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor)

y_pred_scaled = test_predictions.numpy()
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test_tensor.numpy())

#17.计算评价指标

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
print(f"MAE: {mae:.4f} ")
print(f"MSE: {mse:.4f}")
print(f"RMSE:{rmse:.4f} ")
print(f"R²:  {r2:.4f}")

#18.保存预测结果

prediction_result = pd.DataFrame(
    {
        "true_z":y_true.reshape(-1),
        "predicted_z":y_pred.reshape(-1),
        "absolute_error":np.abs(
            y_true.reshape(-1) - y_pred.reshape(-1)
        ),
    }
)

prediction_result.to_csv(
    prediction_path,
    index=False,
)

#19.保存模型参数

torch.save(
    model.state_dict(),
    model_path,
)

#20.预测新数据

new_xy = np.array(
    [
        [
            -1,
            2,
        ]
    ],
    dtype=np.float32,
)

new_x = -1
new_y = 2

if version in [1,2]:

    true_new_z = (np.sin(new_x) + np.cos(new_y))

elif version in [3,4]:

    true_new_z = (np.sin(new_x * new_y) + 0.1 * new_x**2 + np.cos(new_y))

print(
    "真实z:",
    true_new_z
)

new_xy_scaled = X_scaler.transform(new_xy)

new_xy_tensor = torch.tensor(
    new_xy_scaled,
    dtype=torch.float32,
)

model.eval()

with torch.no_grad():
    predicted_z_scaled = model(new_xy_tensor)
    predicted_z = y_scaler.inverse_transform(predicted_z_scaled.numpy())

print(
    "\n预测z:"
    f"{predicted_z.item():.2f}"
)

#21.输出生成文件的位置

print("\n生成的文件：")
print("模拟数据：", data_path.resolve())
print("训练历史：", history_path.resolve())
print("测试预测：", prediction_path.resolve())
print("模型参数：", model_path.resolve())