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
