
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

