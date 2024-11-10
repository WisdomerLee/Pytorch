
import graphlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2'
cars = pd.read_csv(cars_file)
cars.head()

X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)


class LinearRegressionDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)


class LitLinearRegression(pl.LightningModule):
  def __init__(self, input_size, output_size):
    super(LitLinearRegression, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    self.loss_fun = nn.MSELoss()

  def forward(self, x):
    return self.linear(x)

  def configure_optimizers(self):
    learning_rate = 0.02
    optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate) # pytorch와의 차이점 중에 하나로, parameter를 넣을 때 모델 자체가 parameters를 모델의 파라미터로 넘겨줌
    return optimizer

  def training_step(self, train_batch):
    X, y = train_batch
    
    y_pred = self.forward(X)

    loss = self.loss_fun(y_pred, y)
    self.log('Train_loss', loss, prog_bar = True)
    return loss


model = LitLinearRegression(input_size=1, output_size=1)

trainer = pl.Trainer(accelerate='gpu', devices=1, max_epochs=500, log_every_n_steps=2)
trainer.fit(model=model, train_dataloaders=train_loader)

trainer.current_epoch

for parameter in model.parameters():
  print(parameter)

