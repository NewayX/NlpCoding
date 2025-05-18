import numpy as np
import matplotlib.pyplot as plt

# 训练数据
x = np.array([50, 60, 70, 80, 90])  # 面积
y = np.array([150, 180, 210, 240, 270])  # 房价
n = len(x)

# 初始化参数
w = 0.0
b = 0.0
learning_rate = 0.0001
epochs = 2000

# 训练（梯度下降）
for epoch in range(epochs):
    # 预测
    y_pred = w * x + b

    # 计算损失（均方误差）
    loss = np.mean((y - y_pred) ** 2)

    # 计算梯度
    dw = (-2 / n) * np.sum(x * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db

    # 每 200 步打印一次
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

# 4. 画图
plt.scatter(x, y, color='blue', label='真实值')
plt.plot(x, w * x + b, color='red', label='拟合线')
plt.xlabel("面积")
plt.ylabel("价格")
plt.legend()
plt.show()
