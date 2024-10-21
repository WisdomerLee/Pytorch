# 이번엔 하나에 여러 label이 붙을 수 있는 조건으로 분류

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import numpy as np
from collections import Counter

# 데이터 불러오기 
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch - torch.FloatTensor(y)

# 데이터를 훈련, 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.2)

# 데이터 셋 만들기 - 역시 꼭 필요한 __init__, __len__, __getitem__함수를 만들기
class MultilabelDataset(Dataset):
  def __init__(self, X, y):
    self.X=X
    self.y=y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

# 데이터 셋 객체 만들기
multilabel_train_data = MultilabelDataset(X_train, y_train)
multilabel_test_data = MultilabelDataset(X_test, y_test)

# train loader 만들기
train_loader = DataLoader(multilabel_train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(multilabel_test_data, batch_size=32, shuffle=True)
# model class 만들기
class MultilabelNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x) # 여기에서 보면 알겠지만 각 층의 activation function은 항상 같을 필요가 없음, 대체로 feature-extractor의 layer 와 output layer로 가기 전의 activation function은 서로 다르게 쓰는 편
    x = self.fc2(x)
    x = self.sigmoid(x)
    return x

# input, output dim 지정
input_dim = multliabel_train_data.X.shape[1]
output_dim - multilabel_train_data.y.shape[1]

# model 객체 만들기
model = MultilabelNetwork(input_size=input_dim, hidden_size=20, output_size=output_dim)

# loss함수, optimizer 함수 지정하기
loss_fn = nn.BCEWithLogitsLoss() # 해당 함수는 직접 찾아 볼 것 - Binary Cross Entropy 함수와 Log를 결합하여 만든 Loss 함수, 붙일 수 있는 label의 숫자가 단 2개 뿐이므로 Binary 쪽을 이용함

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


losses = []
slope, bias = [], []
number_epochs = 100

for epoch in range(number_epochs):
  
