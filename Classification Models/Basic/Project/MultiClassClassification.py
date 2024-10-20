# 이중 분류가 아닌 다중 분류 모델을 만들어보기

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

# 분류할 데이터 불러오기 - Iris라 불리는 청채꽃??의 종류를 나누는 부분 - 색깔, 모습, 
iris = load_iris()
X= iris.data
y = iris.target

# 데이터를 훈련, 테스트 부류로 나눌 것
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 데이터 형태를 float32로 바꾸기
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# dataset 만들기 - Dataset을 상속받아 꼭 구현해야 하는 3가지 함수를 구현하기 - __init__, __getitem__, __len__
class IrisData(Dataset):
  def __init__(self, X_train, y_train):
    super().__init__()
    self.X = torch.from_numpy(X_train)
    self.y = torch.from_numpy(y_train)
    self.y = self.y.type(torch.LongTensor)
    self.len = self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
  
  def __len__(self):
    return self.len

# dataloader로 변환하기 - dataset을 적당한 크기로 잘려진 반복가능한 형태로 불러오기!!!
iris_data = IrisData(X_train, y_train)
# data를 32개의 크기대로 잘라 한 번에 32개씩의 데이터 묶음을 불러오기, shuffle은 데이터를 원래 순서가 아닌, 임의로 섞은 상태로 32개씩 묶어 불러온다는 것
train_loader = DataLoader(iris_dat, batch_size=32, shuffle=True)

print(f"X shape: {iris_data.X.shape}")

class MultiClassNet(nn.Module):
  # 애초에 분류할 모델이 3가지 밖에 되지 않는 형태이므로, 모델의 deep net도 굉장히 작게 구성됨
  # lin1, lin2, log_softmax

  def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
    super().__init__()
    self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES) # layer를 만들 때 전 layer의 출력부분에 해당되는 숫자가 다음 layer의 입력 부분에 해당되는 숫자와 반드시 일치해야 함을 기억하기
    self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES) # 또한 여기서 사용하는 layer는 선형의 관계를 갖는 형태로 간단한 layer임 - Linear - 마지막 layer의 출력은 반드시 구분되어야 할 종류의 숫자와 일치해야 함!!!
    self.log_softmax = nn.LogSoftmax(dim=1) # 마지막은 항상 activation function을 담당하는 layer를 둘 것 - logsoftmax라는 함수 사용

  def forward(self, x):
    x = self.lin1(x) # 첫 번째 layer 통과
    x = torch.sigmoid(x) # 첫번째 layer의 activation function 통과 - layer 하나를 통과할 때마다 activation function을 지정해야 함
    x = self.lin2(x) # 두 번째 layer 통과
    x = self.log_softmax(x) # 두 번째 layer의 activation function 통과 - layer가 2개이므로 마지막 activation function - 
    return x
    
# 하이퍼 파라미터 지정하기
NUM_FEATURES = iris_data.X.shape[1] # data의 입력 데이터 shape에 맞게 설정해야 함 - 
HIDDEN = 6 # 이것은 layer가 몇 개의 neuron을 가질 것인지를 지정할 파라미터
NUM_CLASSES = len(iris_data.y.unique()) # target이 되는 것의 종류를 이렇게 함수로 넣어주면 데이터에서 클래스의 갯수가 바뀌더라도 하이퍼 파라미터는 그 갯수만큼 자동으로 지정될 것!

# 모델 설정
model = MultiClassNet(NUM_FEATURES=NUM_FEATURES, NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN)

# Loss 함수 설정 - 모델이 가진 구조, 목적에 따라 잘 맞는 loss 함수가 다름!
criterion = nn.CrossEntropyLoss()

LR = 0.01
# optimizer 지정 - 역시 모델이 가진 구조, 목적에 따라 잘 맞는 optimizer 함수가 다름
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# 훈련
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
  for X, y in train_loader:
    # initialize gradients - 먼저 계산된 gradients를 초기화!
    optimizer.zero_grad()
    # forward 모델이 예측
    y_pred_log = model(X)
    
    # loss 값 계산
    loss = criterion(y_pred_log, y)

    # 계산 값 토대로 gradient 변경 값 역전파
    loss.backward()
    # weights 업데이트 - 해당 gradient 변경 값 토대
    optimizer.step()
  losses.append(float(loss.data.detach().numpy())) # loss 값은 gpu에 있는데 이 값을 cpu로 옮기고(detach()) 계산이 빠른 numpy()로 넘김 - 처리

# 손실 값을 그래프로 확인 - 훈련이 제대로 되고 있는지를 확인하기 위해
sns.lineplot(x=range(NUM_EPOCHS), y=losses)

# 
X_test_torch = torch.from_numpy(X_test)
# 아래의 torch.no_grad()는 모델을 gradient 값 계산 없이 forward - 예측만 한다는 것, 평가를 위해서 확인할 때는 꼭 이렇게 함 - 계산 속도가 더 빨라지고, 해당 내용을 토대로 훈련을 진행시킬 것은 아니고 단순 확인용이기 때문
with torch.no_grad():
  y_test_log = model(X_test_log)
  y_test_pred = torch.max(y_test_log.data, 1)

# 얼마나 정확히 예측했는지 확인
accuracy_score(y_test, y_test_pred.indices)

# 
from collections import Counter

most_common_cnt = Counter(y_test).most_common()[0][1]

print(f"Naive classifier accuracy: {most_common_cnt/len(y_test) * 100}%")
# 
