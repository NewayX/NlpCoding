import torch
import numpy as nn


class TorchModel(nn.model):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__(self)
        # 词嵌入层
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # 最大平均池化
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(vector_dim, 1)
        self.activation=torch.sigmoid
        # 均方差
        self.loss=nn.fromfunction.mse_loss

