
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
  def activation(self, x):
    #sigmoid
    return 1/ (1+np.exp(-x))
  
  def dactivation(self, x):
    # derivative of sigmoid
    return self.activation(x) * (1-self.activation(x))

  # forward function
  def forward(self, X):
    hidden_1 = np.dot(X, self.w) + self.b
    activate_1 = self.activation(hidden_1)
    return activate_1
  # backward function
  def backward(self, X, y_true):
    # gradient 계산
    hidden_1 = np.dot(X, self.w) + self.b
    y_pred = self.forward(X)
    dL_dpred 
    
    
