import torch
import torch.nn as nn
import numpy as np
import copy

"""
基于pytorch的网络编写
手动实现梯度计算和反向传播
加入激活函数
"""


class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size, bias=False)  # w = hidden_size * hidden_size  wx+b -> wx
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.layer(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 自定义模型，接受一个参数矩阵作为入参
class DiyModel:
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        x = np.dot(x, self.weight)
        y_pred = self.diy_sigmoid(x)
        if y is not None:
            return self.diy_mse_loss(y_pred, y)
        else:
            return y_pred

    def diy_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def diy_mse_loss(self, y_pred, y):
        return np.sum(np.square(y_pred-y))/len(y_pred)

    def calculate_grad(self,y_pred,y_true,x):
        # loss一阶导数
        grade_mse=2/len(x) * (y_pred - y_true)
        # sigmoid函数 y = 1/(1+e^(-x)) 的导数 = y * (1 - y), 结果为2维向量
        grade_sigmoid=y_pred*(1-y_pred)





