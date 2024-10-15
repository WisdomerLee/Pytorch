import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns

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
X = torch.from_numpy(X_np)
y = torch.tensor(y_list)

# 훈련
w = torch.rand(1, requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

num_epochs=1000 # 훈련을 1천번 시킬 예정이라는 것, 보다 많이 할 수록 훈련용 데이터에 대한 예측이 보다 정교해짐
learning_rate = 0.001 # 파라미터 업데이트 비례 상수 이 값이 크면 파라미터 변동 폭이 크고, 작으면 파라미터 변동 폭이 작음

# batch size를 지정하지 않았는데, 이렇게 되면 1로 잡은 것 - 한 번에 하나의 데이터를 넘기는 형태로... 여러 개의 데이터를 한 번에 묶어서 보내게 되면 처리 시간이 더 짧아짐 - 묶어보낸만큼 병렬로 처리

for epoch in range(num_epochs):
  for i in range(len(X)):
    # forward pass
    y_pred = X[i] * w + b
    # calculate loss
    loss_tensor = torch.pow(y_pred-y[i], 2)

    # backward pass
    loss_tensor.backward()

    # extract losses
    loss_values = loss_tensor.data[0]

    # weight, bias 업데이트
    with torch.no_grad():
      w -= w.grad * learning_rate
      b -= b.grad * learning_rate
      w.grad.zero_() # 뒤에 _가 붙은 함수는 gradient가 이것과 영향을 받는다는 뜻이고, 기본적으로 inplace 작업이라는 것
      b.grad.zero_() # 

  print(loss_value)

# 결과 확인
print(f"Weight: {w.item()}, Bias: {b.item()}")

y_pred = ((X * w) + b).detach().numpy() # torch의 데이터에서 별도로 복사해서 numpy로 넘기기

sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=Xlist, y=y_pred.reshape(-1)) # model을 만들고 훈련할 때 자주 보게 될 것인데, 데이터의 형태는 꼭 확인할 것, 데이터의 형태가 맞지 않아 생기는 오류가 자주 발생할 수 있음

# Linear Regression
from sklearn.linaer_model import LinearRegression
reg = LinearRegression().fit(X_np, y_list)
print(f"Slope: {reg.coef_}, Intercept: {reg.intercept_}")
# linear regression을 이용하여 해당 데이터에 가장 가까운 직선의 기울기와 그 값을 직접 구할 수도 있음!
# graph visualization
# 해당 내용을 실행하기 위해서는 GraphViz라는 library를 설치해야 함 - https://graphviz.org/download/
# 또한 설치 후 재시작이 되지 않을 경우, PATH 변수에 직접 추가할 것 - 실행 위치 등
import os
from torchviz import make_dot
os.environ['PATH'] += os.pathsep + 'C:\Program Files (x86)\Graphviz\bin'
make_dot(loss_tensor)
