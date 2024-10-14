
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import cunfusion_matrix

# 데이터 준비
# 데이터  -  https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv("heart.csv")
df.head()
# 데이터를 변수들과 그 변수에 의존하는 값들로 나누기!!!
X = np.array(df.loc[:, df.columns != 'output']) 
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

# 훈련, 테스트 데이터 나누기 - 아래는 20%의 비율로 테스트 데이터로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 데이터 크기 늘리기, 증폭?
scaler = StandardScaler()
# 
X_train_scale = scaler.fit_transform(X_train) 
X_test_scale = scaler.transform(X_test) 

class NeuralNetworkFromScratch:
  def __init__(self, LR, X_train, y_train, X_test, y_test):
    self.w = np.random.randn(X_train_scale.shape[1])
    self.b = np.random.randn()
    self.LR = LR
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test
    self.L_train = []
    self.L_test = []

  # Helper function
  # activation함수는 모델이 계산한 값을 걸러주는 필터 역할을 수행! > 최종적으로 activation을 통과한 값이 실제 모델이 예측한 값이 됨
  def activation(self, x):
    #sigmoid
    return 1/ (1+np.exp(-x))
  # activation 함수의 도함수(미분함수)
  def dactivation(self, x):
    # derivative of sigmoid
    return self.activation(x) * (1-self.activation(x))

  # forward function - 입력을 갖고 현재 모델의 파라미터를 이용하여 예측 값을 계산
  def forward(self, X):
    hidden_1 = np.dot(X, self.w) + self.b
    activate_1 = self.activation(hidden_1)
    return activate_1
  # backward function - 입력으로 예측을 한 값을 토대로, 실제 값과 예측값의 차이를 이용하여, 그 차이만큼 모델 파라미터를 얼마나 업데이트 시킬 것인지에 대한 값을 계산!
  def backward(self, X, y_true):
    # 예측
    hidden_1 = np.dot(X, self.w) + self.b
    y_pred = self.forward(X)
    
    # gradient 계산
    dL_dpred = 2 * (y_pred-y_true)
    # activation의 gradient함수에 집어넣은 값을 구하고
    dpred_dhidden1 = self.dactivation(hidden1)
    
    dhidden1_db = 1
    dhidden1_dw = X
    # 실제로 파라미터를 얼만큼 업데이트 시킬 것인지를 계산하기
    dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
    dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw
    return dL_db, dL_dw

  # optimizer는 backward에서 계산된 값을 토대로 모델의 파라미터를 업데이트하는 역할을 수행
  def optimizer(self, dL_db, dL_dw):
    # 얼마나 업데이트 시킬 지를 backward 함수에서 계산해두었으므로, 계산된 보정시킬만큼의 값을 파라미터에 업데이트하기
    self.b = self.b - dL_db * self.LR
    self.w = self.w - dL_dw * self.LR

  # 모델 훈련
  def train(self, ITERATIONS):
    for i in range(ITERATIONS):
      # random position
      random_pos = np.random.randint(len(self.X_train))
      # forward pass
      y_train_true = self.y_train[random_pos]
      y_train_pred = self.forward(self.X_train[random_pos])

      # loss 값 계산
      L = np.sum(np.square(y_train_pred-y_train_true))
      self.L_train.append(L)

      # gradient 계산
      dL_db, dL_dw = self.backward(self.X_train[random_pos], self.y_train[random_pos])
      
      # update weights
      self.optimizer(dL_db, dL_dw)

      # test data와의 에러 계산
      L_sum = 0
      for j in range(len(self.X_test)):
        y_true = self.y_test[j]
        y_pred = self.forward(self.X_test[j])
        L_sum += np.square(y_pred-y_true)
      self.L_test.append(L_sum)
    return 'training successful'
    
# 하이퍼 파라미터 지정
LR = 0.1
ITERATIONS = 1000

# 모델 객체 생성 및 훈련
nn = NeuralNetworkFromScratch(LR=LR, X_train=X_train_scale, y_train=y_train, X_test=X_test_scale, y_test=y_test)
nn.train(ITERATIONS=ITERATIONS)

# loss 확인하기
sns.lineplot(x = list(range(len(nn.L_test))), y = nn.L_test)

# 테스트 데이터를 이용하여 반복
total = X_test_scale.shape[0]
correct = 0
y_preds = []
for i in range(total):
  y_true = y_test[i]
  y_pred = np.round(nn.forward(X_test_scale[i]))
  y_preds.append(y_pred)
  correct += 1 if y_true == y_pred else 0

# 정확도 계산
correct / total
# 얼마나 구분해서 뭐가 몇 개 나왔는지 확인하기 - 이것은 이진분류 모델의 기본적인 형태이므로, 0, 1의 값들이 나올 것
from collections import Counter
Counter(y_test)

# confusion matrix로 얼마나 잘못된 값이 있는지 확인
confusion_matrix(y_true=y_test, y_pred = y_preds)
