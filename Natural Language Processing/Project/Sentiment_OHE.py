
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

class SentimentModel(nn.Module):
         def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN=10):
                  super().__init__()
                  self.linear = nn.Linear(NUM_FEATURES, HIDDEN)
                  self.linear2 = nn.Linear(HIDDEN, NUM_CLASSES)
                  self.relu = nn.ReLU()
                  self.log_softmax = nn.LogSoftmax(dim=1)

         def forward(self, x):
                  x = self.linear(x)
                  x = self.relu(x)
                  x = self.linear2(x)
                  x = self.log_softmax(x)
                  return x


model = SentimentModel(NUM_FEATURES=X_train_onehot.shape[1], NUM_CLASSES=3)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
for e in range(NUM_EPOCHS):
         curr_loss = 0
         for X_batch, y_batch in train_loader:
                  optimizer.zero_grad()
                  y_pred_log = model(X_batch)
                  loss = criterion(y_pred_log, y_batch.long())

                  curr_loss += loss.item()
                  loss.backward()
                  optimizer.step()

         train_losses.append(curr_loss)
         print(f"Epoch {e}, Loss: {curr_loss}")


sns.lineplot(x=list(range(len(train_losses))), y=train_losses)
plt.show()


with torch.no_grad():
         for X_batch, y_batch in test_loader:
                  y_test_pred_log = model(X_batch)
                  y_test_pred = torch.argmax(y_test_pred_log, dim=1)

y_test_pred_np = y_test_pred.squeeze().cpu().numpy()

acc = accuracy_score(y_pred=y_test_pred_np, y_true=y_test)

most_common_cnt = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %")

sns.heatmap(confusion_matrix(y_test_pred_np, y_test), annot=True, fmt=".0f")