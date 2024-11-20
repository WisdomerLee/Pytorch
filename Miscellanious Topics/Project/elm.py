# hidden layer 1층으로 되어 학습이 매우 빨라 재학습이 자주 필요할 경우 선택할 수 있는 모델

import numpy as np
from scipy import linalg
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import seaborn as sns
from collections import Counter


X, y = make_blobs(n_samples=10000, n_features=2, centers=5, random_state=1) # 간단한 예시로 만드는 훈련용 데이터 더미!

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sns.scatterplot(X_train[:, 0], X_train[:, 1], hue=y_train) # X의 훈련 데이터 중 첫번째, 두 번째 데이터와 y_train간의 상관관계를 확인

INPUT_SIZE = X_train.shape[1]
HIDDEN_SIZE = 100 # 숨은 layer에 쓰이는 neuron의 갯수!

input_weights = np.random.normal(size=[INPUT_SIZE, HIDDEN_SIZE])
bias = np.random.normal(size=[HIDDEN_SIZE])

def relu(x):
  return np.maximum(x, 0, x) # relu의 정의를 그대로 이용한 함수 > 음수 부분은 0, 양수는 그대로 출력하는 relu함수를 python으로 정의한 것 > 원래는 torch의 내부 함수로 있으나 지금은 torch를 불러오지 않은 상태이므로 relu를 정의하여 쓰는 것

def hidden_nodes(X):
  G = np.dot(X, input_weights) # Linear Algebra의 방식대로 계산하도록 설정 - matrix 곱으로 진행하는 Linear Regression의 형태 그대로
  G = G + biases # bias 보정을 더한 것 > ax+b의 선형 관계를 적용하고 > nn.Linear()와 같은 역할
  H = relu(G) # relu함수 적용 > activation 함수 적용
  return H

# 매우매우 간단한 형태의 관계이기 때문에 연산이 빠름

beta = np.dot(linalg.pinv(hidden_nodes(X_train)), y_train) # output에 적용할 weight 계산 > 이것이 실질적인 모델의 파라미터 역할을 수행
# numpy의 linalg는 선형대수의 함수를 포함하고 있고, pinv함수는 pseudo-inverse matrix를 계산함 > 역행렬과 같은 형태를 하는 함수를 계산해냄 > 해당 것도 역시 torch의 모듈을 이용하면 loss의 backward()라는 함수 내에 포함되어 있는 과정

def predict(X):
  out = hidden_nodes(X)
  out = np.dot(out, beta)
  return out

# 평가!
y_test_pred = predict(X_test)
correct = 0
total = X_test.shape[0]

for i in range(total):
  predicted = np.round(y_test_pred[i], 0)
  y_test_true = y_test[i]
  correct += 1 if predicted == y_test_true else 0

accuracy = correct/total
print(f"Accuracy for {HIDDEN_SIZE} hidden nodes: {accuracy}")

cnt = Counter(y_test)

np.max(list(cnt.values())) / total
