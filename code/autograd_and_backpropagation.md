# 自动求导与反向传播原理

# 1. 自动求导、反向传播和梯度下降的关系

可以把它们理解成：

- **自动求导 Autograd**：自动计算导数或梯度的技术；
- **反向传播 Backpropagation**：一种从损失函数向前面各层反向计算梯度的方法；
- **梯度下降 Gradient Descent**：使用计算出来的梯度更新模型参数。

在 PyTorch 中，最常见的训练代码是：

```python
predictions = model(X)           # 前向传播
loss = loss_fn(predictions, y)   # 计算损失

optimizer.zero_grad()            # 清空旧梯度
loss.backward()                  # 自动求导和反向传播
optimizer.step()                 # 更新模型参数
```

需要特别注意：

> `loss.backward()` 只负责计算梯度，不负责更新参数。

真正更新参数的是：

```python
optimizer.step()
```

---

# 2. 为什么神经网络需要求导

神经网络训练的基本过程是：

1. 使用当前参数进行预测；
2. 计算预测结果和真实结果之间的误差；
3. 判断每个参数对误差产生了什么影响；
4. 修改参数，让误差逐渐减小。

假设有一个最简单的模型：

$$
\hat{y} = wx + b
$$

其中：

* $x$：输入数据；
* $w$：权重；
* $b$：偏置；
* $\hat{y}$：模型预测值。

假设损失函数为平方误差：

$$
L = (\hat{y} - y)^2
$$

为了训练模型，我们需要计算：

$$
\frac{\partial L}{\partial w}
$$

它表示：

> 当权重 $w$ 发生一个很小的变化时，损失 $L$ 会怎样变化。

同样还需要计算：

$$
\frac{\partial L}{\partial b}
$$

它表示偏置 $b$ 对损失的影响。

---

## 2.1 梯度的正负代表什么

如果：

$$
\frac{\partial L}{\partial w} > 0
$$

说明增大 $w$ 会导致损失增大。

因此，为了减小损失，应该减小 $w$。

如果：

$$
\frac{\partial L}{\partial w} < 0
$$

说明增大 $w$ 会导致损失减小。

因此，为了减小损失，应该增大 $w$。

---

## 2.2 梯度下降公式

参数更新公式为：

$$
w_{\text{new}}
=

w_{\text{old}}
-

\eta
\frac{\partial L}{\partial w}
$$

其中：

* $\eta$：学习率；
* $\frac{\partial L}{\partial w}$：损失对权重的梯度。

可以简单记成：

```text
新参数 = 旧参数 - 学习率 × 梯度
```

之所以减去梯度，是因为梯度指向损失增长最快的方向。

反方向就是损失下降的方向。

---

# 3. 什么是计算图

自动求导的核心是：

> 计算图。

假设有下面的计算：

$$
L = (xw + b)^2
$$

计算机会把它拆成多个简单步骤：

$$
u = xw
$$

$$
v = u + b
$$

$$
L = v^2
$$

对应的计算图可以表示为：

```text
x ──┐
    × ── u ──┐
w ──┘        + ── v ── 平方 ── L
         b ──┘
```

每个节点都代表一个运算，例如：

* 乘法；
* 加法；
* 平方；
* 激活函数；
* 矩阵乘法；
* 卷积。

每个节点都需要知道两件事：

1. 前向传播时怎样计算结果；
2. 反向传播时局部导数是多少。

---

## 3.1 乘法节点

假设：

$$
u = xw
$$

那么：

$$
\frac{\partial u}{\partial x} = w
$$

$$
\frac{\partial u}{\partial w} = x
$$

---

## 3.2 加法节点

假设：

$$
v = u + b
$$

那么：

$$
\frac{\partial v}{\partial u} = 1
$$

$$
\frac{\partial v}{\partial b} = 1
$$

---

## 3.3 平方节点

假设：

$$
L = v^2
$$

那么：

$$
\frac{\partial L}{\partial v} = 2v
$$

自动求导框架会记录这些运算，并在反向传播时自动应用对应的求导规则。

---

# 4. 链式法则

链式法则是反向传播的数学基础。

假设：

$$
L = f(v)
$$

$$
v = g(u)
$$

$$
u = h(w)
$$

那么 $w$ 会通过 $u$ 和 $v$ 间接影响 $L$。

链式法则为：

$$
\frac{\partial L}{\partial w}
=

\frac{\partial L}{\partial v}
\cdot
\frac{\partial v}{\partial u}
\cdot
\frac{\partial u}{\partial w}
$$

可以简单理解为：

> 总影响等于每一段局部影响的乘积。

神经网络虽然有很多层，但反向传播的本质就是不断使用链式法则。

---

# 5. 手动完成一次反向传播

考虑：

$$
L = (xw + b)^2
$$

取：

$$
x = 2
$$

$$
w = 3
$$

$$
b = 1
$$

---

## 5.1 前向传播

首先计算：

$$
u = xw
$$

代入数值：

$$
u = 2 \times 3 = 6
$$

然后计算：

$$
v = u + b
$$

代入数值：

$$
v = 6 + 1 = 7
$$

最后计算损失：

$$
L = v^2
$$

$$
L = 7^2 = 49
$$

所以前向传播结果是：

```text
x = 2
w = 3
b = 1
u = 6
v = 7
L = 49
```

---

## 5.2 从损失开始反向传播

首先：

$$
\frac{\partial L}{\partial L} = 1
$$

原因是一个变量对它自己的导数等于 1。

反向传播通常从梯度 1 开始。

---

## 5.3 经过平方运算

因为：

$$
L = v^2
$$

所以：

$$
\frac{\partial L}{\partial v} = 2v
$$

代入 $v = 7$：

$$
\frac{\partial L}{\partial v} = 14
$$

---

## 5.4 经过加法运算

因为：

$$
v = u + b
$$

所以：

$$
\frac{\partial v}{\partial u} = 1
$$

根据链式法则：

$$
\frac{\partial L}{\partial u}
=

\frac{\partial L}{\partial v}
\frac{\partial v}{\partial u}
$$

$$
\frac{\partial L}{\partial u}
=

14 \times 1
=

14
$$

对于偏置 $b$：

$$
\frac{\partial v}{\partial b} = 1
$$

所以：

$$
\frac{\partial L}{\partial b}
=

\frac{\partial L}{\partial v}
\frac{\partial v}{\partial b}
$$

$$
\frac{\partial L}{\partial b}
=

14 \times 1
=

14
$$

---

## 5.5 经过乘法运算

因为：

$$
u = xw
$$

所以：

$$
\frac{\partial u}{\partial w} = x
$$

代入 $x = 2$：

$$
\frac{\partial u}{\partial w} = 2
$$

根据链式法则：

$$
\frac{\partial L}{\partial w}
=

\frac{\partial L}{\partial u}
\frac{\partial u}{\partial w}
$$

$$
\frac{\partial L}{\partial w}
=

14 \times 2
=

28
$$

同理：

$$
\frac{\partial u}{\partial x} = w
$$

代入 $w = 3$：

$$
\frac{\partial L}{\partial x}
=

\frac{\partial L}{\partial u}
\frac{\partial u}{\partial x}
$$

$$
\frac{\partial L}{\partial x}
=

14 \times 3
=

42
$$

最终得到：

$$
\boxed{
\frac{\partial L}{\partial w} = 28
}
$$

$$
\boxed{
\frac{\partial L}{\partial b} = 14
}
$$

$$
\boxed{
\frac{\partial L}{\partial x} = 42
}
$$

反向传播方向是：

```text
L → v → u → w
        ↓
        x
```

前向传播方向是：

```text
输入 → 中间结果 → 损失
```

反向传播方向是：

```text
损失 → 中间结果 → 参数
```

---

# 6. 自动求导到底自动在哪里

如果没有自动求导，我们需要自己推导公式。

例如：

$$
L = (xw + b)^2
$$

对 $w$ 求导：

$$
\frac{\partial L}{\partial w}
=

2(xw + b)x
$$

对 $b$ 求导：

$$
\frac{\partial L}{\partial b}
=

2(xw + b)
$$

对于简单模型，手动求导还比较容易。

但是现实中的神经网络可能包含：

* 数十层甚至数百层网络；
* 数百万或数十亿个参数；
* 卷积层；
* 池化层；
* 激活函数；
* BatchNorm；
* Dropout；
* 残差连接；
* 注意力机制。

如果全部手动求导，会非常困难。

PyTorch 会在前向传播时自动记录：

* 执行了哪些运算；
* 运算的输入是什么；
* 运算的输出是什么；
* 每个运算的求导规则是什么。

调用：

```python
loss.backward()
```

后，PyTorch 会沿着计算图从后向前自动计算梯度。

---

# 7. 自动求导、数值求导和符号求导

## 7.1 数值求导

数值求导使用近似公式：

$$
f'(x)
\approx
\frac{f(x+h)-f(x)}{h}
$$

其中 $h$ 是一个非常小的数。

数值求导的问题包括：

* 只能得到近似结果；
* $h$ 太大会导致误差；
* $h$ 太小会产生浮点数误差；
* 每个参数都要单独计算；
* 参数很多时速度非常慢。

如果模型有一百万个参数，数值求导可能需要进行大量额外计算。

因此，数值求导一般只用于：

> 梯度检查。

它不适合正常训练神经网络。

---

## 7.2 符号求导

符号求导直接推导出导数公式。

例如：

$$
f(x) = x^2 + \sin x
$$

符号求导结果为：

$$
f'(x) = 2x + \cos x
$$

但是复杂模型可能产生非常长的导数表达式，出现表达式膨胀。

---

## 7.3 自动求导

自动求导既不是有限差分近似，也不是生成完整的导数公式。

它的做法是：

1. 把复杂程序拆成基本运算；
2. 保存每个基本运算；
3. 使用每个运算的精确导数；
4. 根据链式法则组合梯度。

因此自动求导具有以下特点：

* 精度高；
* 速度快；
* 适合大量参数；
* 特别适合神经网络训练。

---

# 8. 前向模式和反向模式自动求导

自动求导主要分为：

1. 前向模式自动求导；
2. 反向模式自动求导。

---

## 8.1 前向模式

前向模式会在前向计算过程中，同时传播：

* 数值；
* 导数。

当输入变量很少、输出变量很多时，前向模式比较合适。

例如：

```text
一个输入 → 很多输出
```

---

## 8.2 反向模式

反向模式会先完成前向计算，再从最终输出反向计算所有输入变量的梯度。

神经网络通常有：

* 大量参数；
* 一个标量损失。

例如：

$$
w_1,w_2,w_3,\ldots,w_n
\rightarrow
L
$$

通过一次反向传播，就可以计算：

$$
\frac{\partial L}{\partial w_1}
$$

$$
\frac{\partial L}{\partial w_2}
$$

$$
\cdots
$$

$$
\frac{\partial L}{\partial w_n}
$$

因此，神经网络主要使用反向模式自动求导。

可以认为：

> 反向传播是反向模式自动求导在神经网络中的应用。

---

# 9. 一个真实的模型训练例子

假设模型为：

$$
\hat{y} = wx + b
$$

输入和真实值为：

$$
x = 2
$$

$$
y = 10
$$

初始参数为：

$$
w = 3
$$

$$
b = 1
$$

---

## 9.1 前向传播

预测值为：

$$
\hat{y}
=

3 \times 2 + 1
=

7
$$

损失函数为：

$$
L = (\hat{y} - y)^2
$$

代入：

$$
L = (7 - 10)^2
$$

$$
L = 9
$$

---

## 9.2 计算预测值的梯度

因为：

$$
L = (\hat{y} - y)^2
$$

所以：

$$
\frac{\partial L}{\partial \hat{y}}
=

2(\hat{y} - y)
$$

代入：

$$
\frac{\partial L}{\partial \hat{y}}
=

2(7 - 10)
=

-6
$$

梯度为负数，表示：

> 增大预测值 $\hat{y}$，会让损失减小。

这是合理的，因为当前预测值 7 小于真实值 10。

---

## 9.3 计算权重梯度

因为：

$$
\hat{y} = wx + b
$$

所以：

$$
\frac{\partial \hat{y}}{\partial w} = x
$$

代入 $x = 2$：

$$
\frac{\partial \hat{y}}{\partial w} = 2
$$

根据链式法则：

$$
\frac{\partial L}{\partial w}
=

\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial w}
$$

$$
\frac{\partial L}{\partial w}
=

-6 \times 2
=

-12
$$

---

## 9.4 计算偏置梯度

因为：

$$
\frac{\partial \hat{y}}{\partial b} = 1
$$

所以：

$$
\frac{\partial L}{\partial b}
=

\frac{\partial L}{\partial \hat{y}}
\frac{\partial \hat{y}}{\partial b}
$$

$$
\frac{\partial L}{\partial b}
=

-6 \times 1
=

-6
$$

最终得到：

$$
\boxed{
\frac{\partial L}{\partial w} = -12
}
$$

$$
\boxed{
\frac{\partial L}{\partial b} = -6
}
$$

---

## 9.5 更新参数

假设学习率为：

$$
\eta = 0.1
$$

更新权重：

$$
w_{\text{new}}
=

w
-

\eta
\frac{\partial L}{\partial w}
$$

$$
w_{\text{new}}
=

3
-

0.1 \times (-12)
$$

$$
w_{\text{new}} = 4.2
$$

更新偏置：

$$
b_{\text{new}}
=

b
-

\eta
\frac{\partial L}{\partial b}
$$

$$
b_{\text{new}}
=

1
-

0.1 \times (-6)
$$

$$
b_{\text{new}} = 1.6
$$

使用新参数进行预测：

$$
\hat{y}
=

4.2 \times 2 + 1.6
$$

$$
\hat{y} = 10
$$

在这个简单例子中，模型一步就得到了正确结果。

---

# 10. 对应的 PyTorch 代码

```python
import torch


# ============================================================
# 1. 创建输入和真实值
# ============================================================

# 输入数据通常不需要计算梯度
x = torch.tensor(
    2.0,
)

# 真实标签通常也不需要计算梯度
y = torch.tensor(
    10.0,
)


# ============================================================
# 2. 创建需要训练的参数
# ============================================================

w = torch.tensor(
    3.0,
    requires_grad=True,
)

b = torch.tensor(
    1.0,
    requires_grad=True,
)


# ============================================================
# 3. 前向传播
# ============================================================

y_hat = w * x + b

loss = (y_hat - y) ** 2


print("预测值：")
print(y_hat.item())

print("\n损失：")
print(loss.item())


# ============================================================
# 4. 反向传播
# ============================================================

loss.backward()


print("\nw 的梯度：")
print(w.grad.item())

print("\nb 的梯度：")
print(b.grad.item())


# ============================================================
# 5. 更新参数
# ============================================================

learning_rate = 0.1


with torch.no_grad():

    w -= learning_rate * w.grad

    b -= learning_rate * b.grad


print("\n更新后的 w：")
print(w.item())

print("\n更新后的 b：")
print(b.item())


# ============================================================
# 6. 清空梯度
# ============================================================

w.grad.zero_()

b.grad.zero_()
```

运行结果大约为：

```text
预测值：
7.0

损失：
9.0

w 的梯度：
-12.0

b 的梯度：
-6.0

更新后的 w：
4.2

更新后的 b：
1.6
```

---

# 11. `requires_grad=True` 是什么意思

创建张量时：

```python
w = torch.tensor(
    3.0,
    requires_grad=True,
)
```

其中：

```python
requires_grad=True
```

表示：

> PyTorch 需要追踪所有与 `w` 有关的运算，以便之后计算损失对 `w` 的梯度。

如果没有设置：

```python
requires_grad=True
```

PyTorch 默认不会保存与这个张量有关的求导信息。

例如：

```python
x = torch.tensor(2.0)
```

通常不需要对输入 `x` 求梯度。

训练神经网络时，我们一般修改的是：

* 权重；
* 偏置。

而不是修改输入数据。

因此模型参数通常满足：

```python
weight.requires_grad == True
bias.requires_grad == True
```

---

# 12. 叶子张量和非叶子张量

考虑下面的代码：

```python
w = torch.tensor(
    3.0,
    requires_grad=True,
)

b = torch.tensor(
    1.0,
    requires_grad=True,
)

y_hat = w * x + b

loss = (y_hat - y) ** 2
```

其中：

* `w` 和 `b` 是叶子张量；
* `y_hat` 和 `loss` 是非叶子张量。

---

## 12.1 叶子张量

叶子张量通常是直接创建出来的张量。

例如：

```python
w
b
```

它们不是由其他张量计算得到的。

反向传播完成后，梯度会保存在：

```python
w.grad
b.grad
```

---

## 12.2 非叶子张量

`y_hat` 是通过计算得到的：

```python
y_hat = w * x + b
```

`loss` 也是通过计算得到的：

```python
loss = (y_hat - y) ** 2
```

所以它们是非叶子张量。

默认情况下，PyTorch 通常不会长期保存非叶子张量的 `.grad`。

原因是保存所有中间梯度会占用大量内存。

需要查看非叶子张量梯度时，可以使用：

```python
y_hat.retain_grad()
```

完整示例：

```python
y_hat = w * x + b

y_hat.retain_grad()

loss = (y_hat - y) ** 2

loss.backward()

print(y_hat.grad)
```

---

# 13. `grad_fn` 是什么

可以运行：

```python
print(y_hat.grad_fn)

print(loss.grad_fn)
```

可能会看到：

```text
<AddBackward0 object ...>

<PowBackward0 object ...>
```

这说明：

* `y_hat` 是通过加法运算得到的；
* `loss` 是通过幂运算得到的；
* PyTorch 已经为这些运算建立了反向传播节点。

例如：

```python
y_hat = w * x + b
```

PyTorch 会记录：

1. `w * x` 使用了乘法；
2. 乘法对应的反向求导规则；
3. 乘法结果与 `b` 进行了加法；
4. 加法对应的反向求导规则。

可以把 `grad_fn` 理解成：

> 这个张量是通过什么运算得到的，以及反向传播时应该怎样计算梯度。

叶子张量通常没有 `grad_fn`：

```python
print(w.grad_fn)
```

结果一般为：

```text
None
```

因为 `w` 是直接创建的，不是通过某个运算得到的。

---

# 14. 为什么梯度会累积

PyTorch 中有一个非常重要的规则：

> 每次调用 `backward()`，新计算出的梯度会累加到原来的 `.grad` 中，而不是覆盖原来的梯度。

例如第一次得到：

```text
w.grad = -12
```

如果不清空梯度，下一次又计算得到：

```text
-12
```

那么累积后可能变成：

```text
w.grad = -24
```

因此，每次训练一个 batch 前，一般需要执行：

```python
optimizer.zero_grad()
```

标准训练顺序为：

```python
optimizer.zero_grad()

predictions = model(X_batch)

loss = loss_fn(
    predictions,
    y_batch,
)

loss.backward()

optimizer.step()
```

顺序不能随意改变。

---

## 14.1 为什么 PyTorch 要设计成梯度累积

因为在一些特殊情况下，我们希望多个计算共同影响同一个参数。

例如：

* 一个变量经过多条计算路径；
* 多个小 batch 进行梯度累积；
* 多个损失函数共同训练一个模型。

因此，PyTorch 默认采用梯度累加。

普通训练中，需要主动清零。

---

# 15. 多条路径的梯度为什么要相加

考虑：

$$
y = x^2 + x^3
$$

这里 $x$ 通过两条路径影响 $y$：

```text
        ┌─ 平方 ─┐
x ──────┤        ├─ 加法 ─ y
        └─ 立方 ─┘
```

第一条路径为：

$$
x \rightarrow x^2 \rightarrow y
$$

对应的梯度贡献为：

$$
2x
$$

第二条路径为：

$$
x \rightarrow x^3 \rightarrow y
$$

对应的梯度贡献为：

$$
3x^2
$$

因此总梯度为：

$$
\frac{dy}{dx}
=

2x + 3x^2
$$

所以：

> 当同一个变量通过多条路径影响最终结果时，各条路径传回来的梯度需要相加。

这也是残差网络 ResNet 能够正常反向传播的基础之一。

---

# 16. 多层神经网络如何反向传播

假设一个多层神经网络为：

$$
a^{(0)} = x
$$

第一层：

$$
z^{(1)}
=

W^{(1)}a^{(0)}
+
b^{(1)}
$$

$$
a^{(1)}
=

f^{(1)}(z^{(1)})
$$

第二层：

$$
z^{(2)}
=

W^{(2)}a^{(1)}
+
b^{(2)}
$$

$$
a^{(2)}
=

f^{(2)}(z^{(2)})
$$

最后计算损失：

$$
L = L(a^{(2)}, y)
$$

前向传播过程为：

```text
输入
 ↓
第一层线性变换
 ↓
第一层激活函数
 ↓
第二层线性变换
 ↓
第二层激活函数
 ↓
损失函数
```

反向传播过程为：

```text
损失函数
 ↓
第二层激活函数
 ↓
第二层权重和偏置
 ↓
第一层激活函数
 ↓
第一层权重和偏置
```

---

## 16.1 输出层误差信号

通常定义：

$$
\delta^{(l)}
=

\frac{\partial L}{\partial z^{(l)}}
$$

对于输出层：

$$
\delta^{(L)}
=

\frac{\partial L}{\partial a^{(L)}}
\odot
f'^{(L)}(z^{(L)})
$$

其中：

$$
\odot
$$

表示逐元素相乘。

---

## 16.2 隐藏层误差信号

隐藏层的误差信号为：

$$
\delta^{(l)}
=

\left(
W^{(l+1)}
\right)^T
\delta^{(l+1)}
\odot
f'^{(l)}(z^{(l)})
$$

这条公式可以理解为：

1. 后一层把梯度传回来；
2. 梯度经过后一层权重矩阵的转置；
3. 再乘以当前激活函数的导数。

---

## 16.3 权重和偏置梯度

第 $l$ 层权重的梯度为：

$$
\frac{\partial L}{\partial W^{(l)}}
=

\delta^{(l)}
\left(
a^{(l-1)}
\right)^T
$$

偏置的梯度为：

$$
\frac{\partial L}{\partial b^{(l)}}
=

\delta^{(l)}
$$

这些公式看起来比较复杂，但本质仍然是：

```text
当前节点的梯度
=
上游传来的梯度
×
当前运算的局部导数
```

PyTorch 已经为以下模块写好了反向传播规则：

* `nn.Linear`；
* `nn.Conv2d`；
* `nn.ReLU`；
* `nn.BatchNorm1d`；
* `nn.BatchNorm2d`；
* `nn.MaxPool2d`；
* `nn.CrossEntropyLoss`；
* `nn.MSELoss`。

因此一般不需要手动推导和编写反向传播。

---

# 17. 激活函数在反向传播中的作用

## 17.1 ReLU

ReLU 定义为：

$$
\operatorname{ReLU}(x)
=

\max(0,x)
$$

它的导数为：

$$
\operatorname{ReLU}'(x)
=

\begin{cases}
1, & x > 0 \
0, & x < 0
\end{cases}
$$

因此：

* 输入大于 0 时，梯度可以正常通过；
* 输入小于 0 时，梯度会变成 0。

例如：

```text
上游梯度 = 5
ReLU 输入 = 3
```

由于输入大于 0：

```text
传回梯度 = 5 × 1 = 5
```

如果：

```text
上游梯度 = 5
ReLU 输入 = -3
```

由于输入小于 0：

```text
传回梯度 = 5 × 0 = 0
```

---

## 17.2 Sigmoid

Sigmoid 定义为：

$$
\sigma(x)
=

\frac{1}{1+e^{-x}}
$$

它的导数为：

$$
\sigma'(x)
=

\sigma(x)
\left(
1-\sigma(x)
\right)
$$

Sigmoid 的导数最大只有 0.25。

当输入非常大或非常小时，导数会接近 0。

如果网络很深，多个小于 1 的导数不断相乘，梯度可能越来越小，产生梯度消失。

---

# 18. 梯度消失和梯度爆炸

反向传播需要连续乘以每一层的局部导数。

例如：

$$
\frac{\partial L}{\partial w_1}
=

\frac{\partial L}{\partial a_4}
\frac{\partial a_4}{\partial a_3}
\frac{\partial a_3}{\partial a_2}
\frac{\partial a_2}{\partial a_1}
\frac{\partial a_1}{\partial w_1}
$$

---

## 18.1 梯度消失

如果每一层的导数大约都是：

$$
0.1
$$

经过 10 层后：

$$
0.1^{10}
=

10^{-10}
$$

梯度会变得非常小。

前面的网络层几乎无法得到有效更新。

这就是：

> 梯度消失。

---

## 18.2 梯度爆炸

如果每一层的导数大约都是：

$$
2
$$

经过 10 层后：

$$
2^{10}
=

1024
$$

梯度会变得非常大。

这就是：

> 梯度爆炸。

---

## 18.3 常见解决方法

缓解梯度消失或梯度爆炸的方法包括：

* 使用 ReLU 等激活函数；
* 使用合理的权重初始化；
* 使用 BatchNorm；
* 使用残差连接；
* 使用较合适的学习率；
* 使用梯度裁剪；
* RNN 中使用 LSTM 或 GRU。

梯度裁剪示例：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)
```

---

# 19. 为什么损失通常是标量

通常情况下：

```python
loss.backward()
```

要求 `loss` 是一个标量，也就是只有一个数。

例如：

```python
print(loss.shape)
```

可能输出：

```text
torch.Size([])
```

模型有很多参数，我们希望计算：

$$
\nabla_{\theta} L
$$

其中 $L$ 是一个标量损失。

---

## 19.1 当输出不是标量时

例如：

```python
y = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True,
)
```

这里 `y` 中有三个数。

直接调用：

```python
y.backward()
```

PyTorch 不知道应该怎样组合三个输出。

可以传入一个形状相同的张量：

```python
y.backward(
    torch.ones_like(y)
)
```

在正常神经网络训练中，损失函数一般会使用：

```python
.mean()
```

或者：

```python
.sum()
```

把多个样本的损失合并成一个标量。

例如：

```python
sample_losses = (predictions - y) ** 2

loss = sample_losses.mean()
```

---

# 20. 批量训练时梯度如何计算

假设一个 batch 中有 $N$ 个样本。

每个样本都有一个损失：

$$
L_1,L_2,\ldots,L_N
$$

如果使用平均损失：

$$
L
=

\frac{1}{N}
\sum_{i=1}^{N}L_i
$$

那么梯度为：

$$
\frac{\partial L}{\partial w}
=

\frac{1}{N}
\sum_{i=1}^{N}
\frac{\partial L_i}{\partial w}
$$

也就是说：

> 一个 batch 的梯度，是这个 batch 中所有样本梯度的平均结果。

模型更新时，会综合考虑整个 batch 中的数据，而不是只考虑一个样本。

---

# 21. PyTorch 的动态计算图

PyTorch 默认使用动态计算图。

每次执行：

```python
predictions = model(X)

loss = loss_fn(
    predictions,
    y,
)
```

都会根据这一次实际执行的运算建立计算图。

因此，可以在模型中使用条件判断：

```python
if some_condition:

    output = layer1(x)

else:

    output = layer2(x)
```

PyTorch 会根据本次实际经过的路径建立计算图。

---

## 21.1 为什么不能对同一计算图反向传播两次

通常调用：

```python
loss.backward()
```

后，为了节省内存，PyTorch 会释放计算图中的部分中间信息。

因此再次调用：

```python
loss.backward()
```

可能会出现错误：

```text
Trying to backward through the graph a second time
```

确实需要保留计算图时，可以使用：

```python
loss.backward(
    retain_graph=True,
)
```

但是普通训练一般不需要这样做。

每一轮训练都会重新进行一次前向传播，并重新建立计算图。

---

# 22. 为什么更新参数需要 `torch.no_grad()`

手动更新参数时，一般写成：

```python
with torch.no_grad():

    w -= learning_rate * w.grad
```

原因是参数更新过程：

$$
w_{\text{new}}
=

w
-

\eta
\nabla_w L
$$

不应该被记录到新的计算图中。

如果不关闭梯度追踪，PyTorch 可能会：

* 把参数更新也当成需要求导的运算；
* 导致计算图越来越复杂；
* 产生原地操作错误。

使用优化器时：

```python
optimizer.step()
```

PyTorch 已经在内部正确处理参数更新，因此不需要自己额外添加 `torch.no_grad()`。

---

# 23. `model.eval()` 和 `torch.no_grad()` 的区别

这两个命令经常一起出现，但作用不同。

---

## 23.1 `model.eval()`

```python
model.eval()
```

表示把模型切换到评估模式。

它主要影响：

* Dropout；
* BatchNorm。

例如 Dropout 在训练时会随机丢弃部分神经元，但在评估时不会随机丢弃。

`model.eval()` 并不会自动关闭梯度计算。

---

## 23.2 `torch.no_grad()`

```python
with torch.no_grad():

    predictions = model(X)
```

表示：

> 不建立求导计算图，也不保存反向传播需要的中间结果。

预测时通常同时使用：

```python
model.eval()


with torch.no_grad():

    predictions = model(X)
```

两者区别是：

```text
model.eval()
改变某些网络层的工作方式

torch.no_grad()
关闭自动求导和计算图记录
```

---

# 24. 标准训练循环逐行解释

标准 PyTorch 训练循环为：

```python
for X_batch, y_batch in train_loader:

    optimizer.zero_grad()

    predictions = model(X_batch)

    loss = loss_fn(
        predictions,
        y_batch,
    )

    loss.backward()

    optimizer.step()
```

---

## 24.1 获取一个 batch

```python
for X_batch, y_batch in train_loader:
```

从数据加载器中读取一批输入数据和真实标签。

---

## 24.2 清空旧梯度

```python
optimizer.zero_grad()
```

清空模型参数 `.grad` 中保存的旧梯度。

如果不清空，梯度会不断累加。

---

## 24.3 前向传播

```python
predictions = model(X_batch)
```

模型根据输入计算预测结果。

这个过程中，PyTorch 同时建立计算图。

---

## 24.4 计算损失

```python
loss = loss_fn(
    predictions,
    y_batch,
)
```

比较预测结果和真实结果，得到一个损失值。

回归任务常用：

```python
loss_fn = torch.nn.MSELoss()
```

分类任务常用：

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

---

## 24.5 反向传播

```python
loss.backward()
```

从损失开始，沿着计算图反向计算每个参数的梯度。

梯度会保存在：

```python
parameter.grad
```

例如：

```python
print(
    model.network[0].weight.grad
)

print(
    model.network[0].bias.grad
)
```

---

## 24.6 更新参数

```python
optimizer.step()
```

优化器读取每个参数的梯度，然后更新模型参数。

以最简单的随机梯度下降 SGD 为例：

$$
\theta
\leftarrow
\theta
-

\eta
\nabla_{\theta}L
$$

Adam 和 AdamW 会进行更复杂的处理，但仍然依赖反向传播得到的梯度。

---

# 25. 前向传播时需要保存什么

反向传播需要使用前向传播中的部分中间结果。

例如：

$$
y = x^2
$$

求导为：

$$
\frac{dy}{dx} = 2x
$$

反向传播时需要知道前向传播中的 $x$。

再例如：

$$
y = \operatorname{ReLU}(x)
$$

反向传播时需要知道：

* 前向输入是否大于 0；
* 哪些位置的梯度应该通过；
* 哪些位置的梯度应该变成 0。

因此训练时，PyTorch 会保存部分中间结果，例如：

* 中间激活值；
* 激活函数的输入或输出；
* 卷积层的输入；
* 池化层最大值的位置；
* BatchNorm 的统计信息。

这也是为什么：

> 模型训练通常比模型预测占用更多显存。

预测时使用：

```python
with torch.no_grad():
```

不需要保存反向传播信息，因此显存占用会下降。

---

# 26. CNN 中的反向传播

CNN 的反向传播原理和普通神经网络相同，仍然基于链式法则。

例如：

```python
x = self.conv1(x)

x = self.relu(x)

x = self.pool(x)
```

前向传播顺序为：

```text
输入
→ 卷积层
→ ReLU
→ 池化层
→ 后续网络
→ 损失
```

反向传播顺序为：

```text
损失
→ 后续网络
→ 池化层
→ ReLU
→ 卷积层
→ 输入
```

---

## 26.1 卷积层反向传播

卷积层需要计算：

* 卷积核权重的梯度；
* 偏置的梯度；
* 输入特征图的梯度。

PyTorch 会自动完成这些计算。

---

## 26.2 ReLU 反向传播

ReLU 会根据前向输入的正负决定梯度能否通过。

输入大于 0：

```text
梯度正常通过
```

输入小于 0：

```text
梯度变成 0
```

---

## 26.3 最大池化反向传播

假设最大池化输入为：

```text
1  5
2  3
```

最大值是：

```text
5
```

前向传播输出为：

```text
5
```

如果反向传播时，上游梯度为：

```text
4
```

那么梯度只会传回前向传播中最大值所在的位置：

```text
0  4
0  0
```

其他位置梯度为 0。

原因是池化输出只由最大值 5 决定。

---

# 27. RNN 中的反向传播

RNN 会在多个时间步重复使用同一组参数。

例如：

$$
h_t
=

f(
W_{xh}x_t
+
W_{hh}h_{t-1}
+
b
)
$$

其中：

* $x_t$：当前时间步输入；
* $h_{t-1}$：上一个时间步的隐藏状态；
* $h_t$：当前隐藏状态；
* $W_{xh}$：输入到隐藏层的权重；
* $W_{hh}$：隐藏状态到隐藏状态的权重。

RNN 的反向传播称为：

> 随时间反向传播，Backpropagation Through Time，简称 BPTT。

它会把 RNN 按时间展开：

```text
x1 → h1 → h2 → h3 → h4
      ↑     ↑     ↑     ↑
     x1    x2    x3    x4
```

然后从最后一个时间步开始反向传播：

```text
损失
← h4
← h3
← h2
← h1
```

由于同一个权重会在多个时间步重复使用，因此它收到的梯度需要相加。

RNN 容易出现梯度消失和梯度爆炸，是因为梯度需要跨越很多时间步连续相乘。

LSTM 和 GRU 的重要作用之一，就是缓解普通 RNN 的长期梯度问题。

---

# 28. 自动求导代码观察实验

下面的代码可以帮助观察 PyTorch 自动求导过程：

```python
import torch


x = torch.tensor(
    2.0,
)

w = torch.tensor(
    3.0,
    requires_grad=True,
)

b = torch.tensor(
    1.0,
    requires_grad=True,
)


u = x * w

v = u + b

loss = v ** 2


print("w 是否需要梯度：")
print(w.requires_grad)

print("\nb 是否需要梯度：")
print(b.requires_grad)

print("\nu 的 grad_fn：")
print(u.grad_fn)

print("\nv 的 grad_fn：")
print(v.grad_fn)

print("\nloss 的 grad_fn：")
print(loss.grad_fn)


loss.backward()


print("\nw 的梯度：")
print(w.grad)

print("\nb 的梯度：")
print(b.grad)
```

理论结果为：

$$
\frac{\partial L}{\partial w} = 28
$$

$$
\frac{\partial L}{\partial b} = 14
$$

所以程序输出应接近：

```text
w 的梯度：
tensor(28.)

b 的梯度：
tensor(14.)
```

---

# 29. 查看模型参数梯度

假设模型为：

```python
import torch
import torch.nn as nn


model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
```

可以查看模型参数名称：

```python
for name, parameter in model.named_parameters():

    print(name)

    print(parameter.shape)
```

在反向传播之前：

```python
for name, parameter in model.named_parameters():

    print(name)

    print(parameter.grad)
```

通常会看到：

```text
None
```

因为还没有进行反向传播。

完成前向传播和反向传播：

```python
X = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)

y = torch.tensor(
    [
        [5.0],
        [9.0],
    ]
)


predictions = model(X)

loss = torch.mean(
    (predictions - y) ** 2
)

loss.backward()
```

再次查看：

```python
for name, parameter in model.named_parameters():

    print(name)

    print(parameter.grad)
```

此时每个参数一般都有对应的梯度。

---

# 30. 自动求导常见问题

## 30.1 为什么 `.grad` 是 `None`

可能原因包括：

1. 没有设置 `requires_grad=True`；
2. 还没有调用 `backward()`；
3. 查看的是非叶子张量；
4. 使用了 `torch.no_grad()`；
5. 中途使用了 `.detach()`；
6. 张量没有参与最终损失计算。

---

## 30.2 为什么梯度越来越大

可能原因包括：

* 没有执行 `optimizer.zero_grad()`；
* 梯度发生了累积；
* 学习率过大；
* 出现了梯度爆炸；
* 数据没有标准化；
* 模型初始化不合理。

---

## 30.3 为什么梯度全是 0

可能原因包括：

* ReLU 输入全部小于 0；
* 使用了不合适的激活函数；
* 网络过深，出现梯度消失；
* 参数没有参与损失计算；
* 某个地方使用了 `.detach()`；
* 在 `torch.no_grad()` 中完成了前向传播。

---

## 30.4 为什么不能直接修改参数

下面的代码可能产生错误：

```python
w -= learning_rate * w.grad
```

因为 `w` 是需要梯度的叶子张量，直接进行原地修改可能影响计算图。

正确方式是：

```python
with torch.no_grad():

    w -= learning_rate * w.grad
```

或者直接使用：

```python
optimizer.step()
```

---

# 31. `.detach()` 是什么

`.detach()` 会返回一个与原张量共享数据，但不再参与当前计算图的新张量。

例如：

```python
new_tensor = old_tensor.detach()
```

之后使用 `new_tensor` 进行计算，不会把梯度传回 `old_tensor`。

常见用途包括：

* 将张量转换成 NumPy；
* 阻断某一部分梯度；
* RNN 中截断计算图。

例如：

```python
predictions_numpy = (
    predictions
    .detach()
    .cpu()
    .numpy()
)
```

如果张量需要梯度，通常不能直接调用：

```python
predictions.numpy()
```

需要先：

```python
predictions.detach()
```

---

# 32. `item()` 是否会影响梯度

```python
loss.item()
```

会把只包含一个数的张量转换成普通 Python 数值。

例如：

```python
print(
    loss.item()
)
```

它常用于打印或记录损失。

但是 `item()` 得到的是普通 Python 数值，不再属于计算图。

因此不能写：

```python
loss_value = loss.item()

loss_value.backward()
```

因为普通数字没有 `backward()` 方法。

正确方式是：

```python
loss.backward()

print(
    loss.item()
)
```

---

# 33. 自动求导和优化器的分工

自动求导系统负责：

```text
计算每个参数的梯度
```

优化器负责：

```text
根据梯度更新参数
```

例如：

```python
loss.backward()
```

执行后，参数中保存：

```python
parameter.grad
```

随后：

```python
optimizer.step()
```

优化器读取这些梯度并更新参数。

---

## 33.1 SGD

最简单的 SGD 更新公式为：

$$
\theta
\leftarrow
\theta
-

\eta
\nabla_{\theta}L
$$

---

## 33.2 Adam

Adam 不只是使用当前梯度，还会记录：

* 梯度的一阶移动平均；
* 梯度平方的二阶移动平均。

它会为不同参数自动调整更新幅度。

---

## 33.3 AdamW

AdamW 在 Adam 的基础上，更合理地处理权重衰减。

但无论使用哪种优化器，基础信息都来自：

```python
loss.backward()
```

计算得到的梯度。

---

# 34. 完整训练循环示例

```python
import torch
import torch.nn as nn


# ============================================================
# 1. 准备数据
# ============================================================

X = torch.tensor(
    [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ],
    dtype=torch.float32,
)

y = torch.tensor(
    [
        [3.0],
        [5.0],
        [7.0],
        [9.0],
    ],
    dtype=torch.float32,
)


# ============================================================
# 2. 创建模型
# ============================================================

model = nn.Linear(
    in_features=1,
    out_features=1,
)


# ============================================================
# 3. 创建损失函数
# ============================================================

loss_fn = nn.MSELoss()


# ============================================================
# 4. 创建优化器
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)


# ============================================================
# 5. 训练模型
# ============================================================

epochs = 1000


for epoch in range(epochs):

    # 清空上一轮梯度
    optimizer.zero_grad()

    # 前向传播
    predictions = model(X)

    # 计算损失
    loss = loss_fn(
        predictions,
        y,
    )

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch {epoch + 1:4d}, "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# 6. 查看训练后的参数
# ============================================================

print("\n训练后的权重：")

print(
    model.weight.item()
)

print("\n训练后的偏置：")

print(
    model.bias.item()
)
```

数据规律为：

$$
y = 2x + 1
$$

训练完成后，模型参数应该接近：

```text
权重 w ≈ 2
偏置 b ≈ 1
```

---

# 35. 自动求导和反向传播的完整流程

整个神经网络训练过程可以概括为：

## 第一步：前向传播

```text
输入数据
→ 网络各层
→ 预测结果
→ 损失函数
→ 得到损失
```

数学形式：

$$
x
\rightarrow
\hat{y}
\rightarrow
L
$$

---

## 第二步：建立计算图

PyTorch 在前向传播时自动记录：

* 每一步执行的运算；
* 运算之间的连接关系；
* 反向传播需要的中间结果；
* 每个运算对应的求导规则。

---

## 第三步：反向传播

执行：

```python
loss.backward()
```

从损失函数出发，按照计算图反向应用链式法则。

得到：

$$
\frac{\partial L}{\partial w_1},
\frac{\partial L}{\partial w_2},
\ldots,
\frac{\partial L}{\partial b_1},
\frac{\partial L}{\partial b_2}
$$

---

## 第四步：保存梯度

梯度保存在参数的：

```python
parameter.grad
```

中。

---

## 第五步：更新参数

执行：

```python
optimizer.step()
```

根据梯度更新参数。

---

## 第六步：重复训练

使用更新后的参数重新进行：

```text
前向传播
→ 计算损失
→ 反向传播
→ 更新参数
```

经过很多轮训练后，损失逐渐减小，模型预测逐渐准确。

---

# 36. 最重要的知识总结

## 36.1 自动求导

自动求导会：

* 记录前向传播中的运算；
* 构建计算图；
* 保存求导需要的中间信息；
* 自动应用链式法则；
* 计算模型参数的梯度。

---

## 36.2 反向传播

反向传播会：

* 从最终损失开始；
* 按照计算图反向移动；
* 计算每个节点的局部导数；
* 将上游梯度乘以局部导数；
* 多条路径的梯度进行相加；
* 最终得到每个参数对损失的梯度。

---

## 36.3 梯度下降

梯度下降会：

* 读取参数梯度；
* 根据学习率决定更新幅度；
* 沿损失减小的方向修改参数。

---

# 37. 最核心的五句话

1. **前向传播负责计算预测值和损失。**

2. **计算图记录前向传播中执行的每一步运算。**

3. **反向传播从损失出发，按照链式法则反向计算梯度。**

4. **`loss.backward()` 只计算梯度，并把梯度保存在参数的 `.grad` 中。**

5. **`optimizer.step()` 才会真正更新模型参数。**

整个训练过程可以记成：

```text
前向传播：

参数
→ 预测值
→ 损失
```

```text
反向传播：

损失
→ 梯度
→ 每个模型参数
```

```text
参数更新：

旧参数
-
学习率 × 梯度
=
新参数
```

---

# 38. 一句话总结

> 反向传播并不是让神经网络倒着运行，而是从最终损失出发，沿着前向传播建立的计算图反向应用链式法则，计算每个参数对损失的影响程度，再由优化器根据这些梯度更新参数。
**
