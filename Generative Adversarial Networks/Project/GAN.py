
import torch
from torch.utils.data import DataLoader
from torch import nn

import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import seaborn as sns

sns.set(rc={'figure.figsize': (12,12)})

# 훈련 데이터 만들기
TRAIN_DATA_COUNT=1024

theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)])

x = 16 * (np.sin(theta) ** 3)
y = 13 * np.cos(theta)- 5 * np.cos(2*theta) - 2 * np.cos(3* theta) - np.cos(4*theta)

sns.scatterplot(x=x, y=y)

train_data = torch.Tensor(np.stack((x,y), axis=1))
train_labels = torch.zeros(TRAIN_DATA_COUNT)
train_set = [
  (train_data[i], train_labels[i]) for i in range(TRAIN_DATA_COUNT)
]

# dataloader
BATCH_SIZE = 64
train_loader = DataLoader(
  train_set, batch_size=BATCH_SIZE, shuffle=True
)
