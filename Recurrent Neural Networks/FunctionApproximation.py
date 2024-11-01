#
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

# 시간에 따라 어떻게 바뀌는지를, 그리고, 그 값이 어떻게 변화할지를 예측하는데에
num_points = 360*4
X = np.arange(num_points)
y = [np.cos(X[i]*np.pi/180)* (1+i/num_points) + (np.random.rand())]
sns.lineplot(x=X, y=y)

# 데이터를 작은 조각들로 보내기 위해 아래와 같이 데이터를 재구성
X_restruct = []
y_restruct = []

# 아래의 과정은 10개의 포인트씩 묶어서 np의 배열로 재구성하고, 
for i in range(num_points-10):
  list1 = []
  for j in range(i, i+10):
    list1.append(y[j])
  X_restruct.append(list1)
  y_restruct.append(y[j+1])
X_restruct = np.array(X_restruct)
y_restruct = np.array(y_restruct)

# 재구성된 데이터를 훈련용, 테스트용으로 나누기
train_test_clipping = 360*3
X_train = X_restruct[:train_test_clipping]
X_test = X_restruct[train_test_clipping:]
y_train = y_restruct[:train_test_clipping]
y_test = y_restruct[train_test_clipping:]

#Dataset
class TrigonometricDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

train_loader = DataLoader(TrigonometricDataset(X_train, y_train), batch_size=,)
test_loader = DataLoader(TrigonometricDataset(X_test, y_test), batch_size=,)

sns.lineplot(x=range(len(y_train)), y=y_train, label = 'Train Data')

class TrigonometryModel(nn.Module):
  def __init__(self, input_size=1, output_size=1):
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size=5, num_layers=1, batch_first=True)
    self.fc1 = nn.Linear(in_features=5, out_features=output_size)
    self.relu = nn.ReLU() # ReLU를 사용하면 값이 0이하의 경우로 내려갈 수 없는 문제가 있음! -> 그렇기 때문에 activation function을 다른 것으로 바꾸어야 할 필요가 있는데, 

  def forward(self, x):
    x, status = self.lstm(x) # Bs, Seq_len, hidden
    x = x[:, -1, :]
    x = self.fc1(x)
    x = self.relu(x) # 해당 내용을 적용하지 않거나... 혹은....
    return x
  

    
    
model = TrigonometryModel()

# 아래의 주석은 forward로 layer를 추가할 때 output을 확인하여 data shape에 맞게 입력할 수 있도록 설정하는 것
# input = torch.rand((2, 10, 1))
# model(input).shape

loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
NUM_EPOCHS = 200

for epoch in range(NUM_EPOCHS):
  for j, (X, y) in enumerate(train_loader):
    optimizer.zero_grad()

    y_pred = model(X.view(-1, 10, 1)) # train의 shape을 확인하고, y의 출력 datashape과 비교하여 다음과 같이 조정

    loss = loss_fun(y_pred, y.unsqueeze(1))
    loss.backward()
    optimizer.step()
  if epoch % 50 == 0:
    print(f"Epoch: {epoch}, Loss: {loss.data}")


test_set = TrigonometricDataset(X_test, y_test)
X_test_torch, y_test_torch = next(iter(test_loader))

with torch.no_grad():
  y_pred = model(torch.unsqueeze(X_test_torch, 2)).detach().squeeze().numpy()

y_act = y_test_torch.numpy()
x_act = range(y_act.shape[0])

sns.lineplot(x=x_act, y=y_act, label = 'Actual', color='black')
sns.lineplot(x=x_act, y=y_pred, label = 'Predicted', color='red')

sns.scatterplot(x=y_act, y=y_pred, label = 'Predicted', color='red', alpha=0.5)

