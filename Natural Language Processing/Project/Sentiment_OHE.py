
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import seaborn as sns

twitter_file = 'data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
df

# 분석하게 될 감정
cat_id = {'neutral': 1,
         'negative': 0,
         'positive': 2}

df['class'] = df['sentiment'].map(cat_id) # 읽어들인 데이터에 sentiment를 cat_id로 변환시킨 class 요소를 추가함!

BATCH_SIZE = 512
NUM_EPOCHS = 80

X = df['text'].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.5, random_state=123
)
print(f"X train: {X_train.shape}, y train: {y_train.shape}\n X test: {X_test.shape}, y test: {y_test.shape}")

one_hot = CountVectorizer()
X_train_onehot = one_hot.fit_transform(X_train)
X_test_onehot = one_hot.fit_transform(X_test)

class SentimentData(Dataset):

  def __init__(self, X, y):
    super().__init__()
    self.X = torch.Tensor(X.toarray())
    self.y = torch.Tensor(y).type(torch.LongTensor)
    self.len = len(self.x)

  def __len__(self):
    return self.len

  def __getitem(self, index):
    return self.X[index], self.y[index]

train_ds = SentimentData(X=X_train_onehot, y=y_train)
test_ds = SentimentData(X=X_test_onehot, y=y_test)

train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=15000)
