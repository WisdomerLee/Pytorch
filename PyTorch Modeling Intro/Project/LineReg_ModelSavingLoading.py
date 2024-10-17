import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

# 분석할 기본 데이터 불러오기 - csv 파일
cars_file = ''
cars = pd.read_csv(cars_file)
cars.head()

# 그래프로 데이터 확인하기
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

# 데이터를 tensor로 변환하기
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1) # 
y_list = cars.mpg.values.tolist()
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y = torch.tensor(y_list)

# dataset, dataloader 만들기
# 사용자 지정 dataset을 만들려면 아래와 같이 기본적인 3개의 함수는 반드시 구현해두어야 함 - __init__(self, X, y), __len__(self), __get_item__(self, idx)
class LinearRegressionDataset(Dataset):
  
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)

# 모델 클래스
class LinearRegressionTorch(nn.Module):
  def __init__(self, input_size, output_size):
    super(LinearRegressionTorch, self).__init__()
    self.linear = nn.Linear(input_size, output_size)

  def forward(self, x):
    out = self.linear(x)
    return out

input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_dim, output_dim)
model.train()

# Loss 함수
loss_fun = nn.MSELoss() # torch에서 미리 지정된 분산 함수 사용

# optimizer
LR = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr=LR) # optimizer함수는 대체로 모델의 파라미터와 learning rate를 입력해야 함, torch의 모든 모델들은 parameters()라는 함수로 모델의 모든 파라미터를 얻을 수 있음

# train loader 확인
for i, (X, y) in enumerate(train_loader):
  print("{i}th batch \n{X}\n{y}")


# 훈련
losses, slope, bias = [], [], []

NUM_EPOCHS = 1000
BATCH_SIZE = 2 # 데이터를 한 번에 보내지 않고 이 크기 단위로 쪼개서 보내기!를 설정
for epoch in range(NUM_EPOCHS):
  for i, (X, y) in enumerate(train_loader):
    # 기존에 계산했던 gradient 값을 비우고
    optimizer.zero_grad()
    # 모델이 예측하면
    y_pred = model(X) # dataloader는 batch_size만큼씩 데이터를 쪼갠 형태로 데이터를 불러오기 때문에 데이터의 위치, batch_size등을 훈련 루프에서 고려할 필요가 없음!
    # loss 값 계산
    loss = loss_fun(y_pred, y) # 이 역시 마찬가지
    # gradient 계산!!
    loss.backward()
  
    # weights 업데이트
    optimizer.step()

    # parameters 얻기
    for name, param in model.named_parameters():
      if param.requires_grad:
        if name == 'linear.weight':
          slope.append(param.data.numpy()[0][0])
        if name == 'linear.bias':
          bias.append(param.data.numpy()[0])
    losses.append(float(loss.data))

    if epoch % 100 == 0:
      print("Epoch: {}, Loss{:.4f}".format(epoch, loss.data))

# 모델의 state_dict

# 모델의 state_dict 저장

# 모델 파라미터 불러오기

# 
