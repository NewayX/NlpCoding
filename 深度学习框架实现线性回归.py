import torch
import torch.nn as nn
import matplotlib.pylab as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False               # 解决负号显示问题

x=torch.tensor([[50.0],
                [60.0],
                [70.0],
                [80.0],
                [90.0]])
y = torch.tensor([[150.0],
                  [180.0],
                  [210.0],
                  [240.0],
                  [270.0]])

model=nn.Linear(in_features=1,out_features=1)

loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.0001)

for epoch in range(2000):

    y_pred=model(x)

    # 计算损失（均方误差）
    loss=loss_fn(y_pred,y)

    # 计算梯度
    optimizer.zero_grad()
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 200 ==0:
        print(f"Epoch{epoch},Loss:{loss.item():.4f}")

w,b=model.parameters()
print(f"Learned weight:{w.item():.4f},bias:{b.item():.4f}")

# 6. 画图
predicted = model(x).detach().numpy()
plt.scatter(x.numpy(), y.numpy(), label='真实值')
plt.plot(x.numpy(), predicted, 'r-', label='预测线')
plt.xlabel("面积")
plt.ylabel("价格")
plt.legend()
plt.show()



