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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 해당 함수 역시 마찬가지- loss의 gradient를 이용하여 weight를 어떻게 업데이트 시키는지는 함수에 따라 달라짐, 또한 모델의 파라미터를 모두 정보를 갖고 있어야 하는데, 모든 nn.Module을 상속받은 모델들은 parameters()라는 함수로 모델이 가진 파라미터들을 확인할 수 있음

losses = []
slope, bias = [], []
number_epochs = 100

for epoch in range(number_epochs):
  for j, (X, y) in enumerate(train_loader):
    # optimizer에 먼저 계산된 gradient를 초기화
    optimizer.zero_grad()

    # 모델이 현재 가진 파라미터로 예측
    y_pred = model(X)

    # 손실 정도 계산
    loss = loss_fn(y_pred, y)
    # 손실된 정도를 바탕으로 parameter별 gradient 계산, 적용
    loss.backward()
    # weight 업데이트
    optimizer.step()
  # 진행률을 확인하기 위해 만들어 두기 - 10 단위씩 묶어서 손실 값이 얼마나 줄어드는지 확인하기 - 훈련이 제대로 되는지 확인하기 위해 필요 - 만약 이 값이 줄어들지 않는다면, 모델의 구조, 함수 등을 바꾸어 적용해야 함
  if epoch % 10 == 0:
    print(f'epoch:{epoch}, loss: {loss.data.item()}')
    losses.append(loss.item())

# loss 값 변화 추적
sns.scatterplot(x=range(len(losses)), y=losses)

# test 데이터로 모델 평가하기
with torch.no_grad():
  y_test_pred = model(X_test).round()


y_test_str = [str(i) for i in y_test.detach().numpy()]

most_common_cnt = Counter(y_test_str).most_common()[0][1]

print("Naive Classifier accuracy: {most_common_cnt/len({y_test_str})*100}%")

# 정확도 측정
accuracy_score(y_test, y_test_pred)
