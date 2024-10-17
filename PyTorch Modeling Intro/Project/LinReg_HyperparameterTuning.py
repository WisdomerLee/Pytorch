import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from torch.utils.data import Dataset, DataLoader

from skorch import NueralNetRegressor
from sklearn.model_selection import GridSearchCV


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

# 이번엔 연구 쪽에 가까운데, 모델을 두고 어느 하이퍼 파라미터들의 조합이 가장 좋은지를 찾아가는 것
net = NeuralNetRegressor(
  LineRegressionTorch,
  max_epochs=100,
  lr=LR,
  iteration_train__shuffle=True,
)

net.set_params(train_split=False, verbose=0)
# 조절할 파라미터의 이름과 그 파라미터가 가질 수 있는 값을 아래와 같이 지정하고
params = {
  'lr': [0.02, 0.05, 0.08 ],
  'max_epochs': [10, 200, 500],  
}
# 아래와 같이 GridSearch를 진행할 수 있도록 설정
gs = GridSearchCV(net, params, scoring='neg_mean_squared_error', cv=3, verbose=2)

# 최적의 조합 찾기는 아래의 한줄의 함수로 진행된다...! 매우 간단
gs.fit(X, y_true)

# 찾은 파라미터 조합과 최고 점수는 아래와 같이 확인 가능
print(f"Best score : {gs.best_score_}, Best params: {gs.best_params_}")

