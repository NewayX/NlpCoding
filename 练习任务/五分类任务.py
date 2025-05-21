# 五个输入，哪个数最大属于第几类，损失函数用交叉墒
import numpy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，五个输入，哪个数最大属于第几类，损失函数用交叉墒

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层，5分类
        self.loss = nn.CrossEntropyLoss()  # loss函数使用交叉熵

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size,input_size)->(batch_size,5)

        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            y_logits = torch.softmax(y_pred, dim=1)
            predicted_class = torch.argmax(y_logits, dim=1)  # 取输出的五维logits的最大值的索引
            return predicted_class  # 输出预测结果


def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, max_index


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def main():
    model = TorchModel(5)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    x, y = build_dataset(100)
    # print("输入：", x)
    # print("标签：", y)
    #
    # input_vecs = [
    #     [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #     [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    # ]
    #
    # tensor_input = torch.FloatTensor(input_vecs)  # shape: [2, 5]
    # result = model(tensor_input)
    #
    # print("result是", result)
    for epoch in range(1000):
        loss = model(x, y)

        loss.backward()

        optim.step()
        optim.zero_grad()

    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    return


def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d " % (vec, res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [
        [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843], #4
        [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],#2
        [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],#1
        [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]#2
    ]
    predict("model.bin", test_vec)
