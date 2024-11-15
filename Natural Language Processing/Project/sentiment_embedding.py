
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer # 단어 단위가 아니라 문장 단위로 변환할 수 있는 방식, 활용을 위해서는 별도의 설치가 필요 pip install sentence-transformers

twitter_file = 'data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
df

cat_id = {'neutral': 1,
         'negative': 0,
         'positive': 2}

df['class'] = df['sentiment'].map(cat_id)

BATCH_SIZE=128
NUM_EPOCHS=10
MAX_FEATURES=10

emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1') # hugging face에 있는 모델 중 하나

sentences = ["Each sentence is converted"]
embeddings = embed_model.encode(sentences)
print(embeddings.squeeze().shape)

# X = emb_model.encode(df['text'].values)
# with open("data/tweets_X.pkl", "wb") as output_file:
#   pickle.dump(X_output_file)

with open("data/tweets_X.pkl", "rb") as input_file:
  X = pickle.load(input_file)

y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

class SentimentData(Dataset):
  def __init__(self, X, y):
    super().__init__()
    self.X = torch.Tensor(X)
    self.y = torch.Tensor(y).type(torch.LongTensor)
    self.len = len(self.X)

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return self.X[index], self.y[index]

train_ds = SentimentData(X=X_train, y=y_train)
test_ds = SentimentData(X_test, y_test)

train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=15000)

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

model = SentimentModel(NUM_FEATURES=X_train.shape[1], NUM_CLASSES=3)

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

with torch.no_grad():
  for X_batch, y_batch in test_loader:
    y_test_pred_log = model(X_batch)
    

    
