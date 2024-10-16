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

# Loss 함수
loss_fun = nn.MSELoss() # torch에서 미리 지정된 분산 함수 사용

# optimizer
LR = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr=LR) # optimizer함수는 대체로 모델의 파라미터와 learning rate를 입력해야 함, torch의 모든 모델들은 parameters()라는 함수로 모델의 모든 파라미터를 얻을 수 있음

# 훈련
losses, slope, bias = [], [], []

NUM_EPOCHS = 1000
BATCH_SIZE = 2 # 데이터를 한 번에 보내지 않고 이 크기 단위로 쪼개서 보내기!를 설정
for epoch in range(NUM_EPOCHS):
  for i in range(0, X.shape[0], BATCHSIZE):
    # 기존에 계산했던 gradient 값을 비우고
    optimizer.zero_grad()
    # 모델이 예측하면
    y_pred = model(X[i:i + BATCH_SIZE]) # 그래서 데이터의 전체가 아닌 데이터의 batch_size만큼만 취해서 보내기
    # loss 값 계산
    loss = loss_fun(y_pred, y_true[i: i+ BATCH_SIZE]) # 역시 비교할 대상도 데이터의 같은 위치 batch_size의 크기만큼 취해서 비교
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
  
    
# 모델이 훈련한 것을 그림으로 확인하기
sns.scatterplot(x=range(NUM_EPOCHS), y=losses)

# bias 값을 추적해보기
sns.scatterplot(x=range(NUM_EPOCHS), y=bias)
# slope 값을 추적해보기
sns.scatterplot(x=range(NUM_EPOCHS), y=slope)
# 결과 확인하기

y_pred = model(X).data.numpy().reshape(-1) # 맨 앞의 차원 하나를 지우기 > 데이터의 타입을 입력에 맞게 바꾸기 위함
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')

# learning_rate와 NUM_EPOCHS는 실험을 반복하며 loss 값이 0 근처로 떨어지는지를 확인해야 함
