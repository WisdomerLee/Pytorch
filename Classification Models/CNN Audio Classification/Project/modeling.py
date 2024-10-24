
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

transform = transforms.Compose([
  transforms.Resize((100, 100)),
  transforms.Grayscale(num_output_channels=1),
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
test_loader= torch.utils.data.DataLoader(testset, batch_size, batch_size, shuffle=True)

CLASSES = ['artifact', 'extrahls', 'murmur', 'normal']
NUM_CLASSES = len(CLASSES)
class ImageMulticlassClassificationNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(100*100, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, NUM_CLASSES)
    self.relu = nn.ReLU()
    self.softmax = nn.LogSoftmax()

  def forward(self, x):
    x = self.conv1(x) # out: (BS, 6, 100, 100)
    x = self.relu(x)
    x = self.pool(x) # out: (BS, 6, 50, 50)
    x = self.conv2(x) # out: (BS, 16, 50, 50)
    x = self.relu(x)
    x = self.pool(x) # out: (BS, 16, 25, 25)
    x = self.flatten(x) # out: (BS, 10000) 16*25*25
    x = self.fc1(x) # out: (BS, 128)
    x = self.relu(x)
    x = self.fc2(x) # out: (BS, 64)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.softmax(x)
    return x

# input = torch.rand(1,1, 100, 100) # layer 계산을 위해 추가된 코드
model = ImageMulticlassClassificationNet()

# model(input).shape # layer 계산을 위해 추가된 코드

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

losses_epoch_mean = []
NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
  losses_epoch = []
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = model(inputs)

    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    losses_epoch.append(loss.item())
  losses_epoch_mean.append(np.mean(losses_epoch))
  print(f"Epoch {epoch}/{NUM_EPOCHS}, Loss {np.mean(losses_epoch_mean)}")

sns.lineplot(x=list(range(len(losses_epoch_mean))), y=losses_epoch_mean)

y_test = []
y_test_hat = []
for i, data in enumerate(test_loader, 0):
  inputs, y_test_hat = data
  with torch.no_grad():
    y_test_hat_temp = model(inputs).round()
  y_test.extend(y_test_temp.numpy())
  y_test_hat.extend(y_test_hat_temp.numpy())

Counter(y_test)

acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f}%')

cm = confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
sns.heatmap(cm, annot=True, xticklabels=CLASSES, yticklabels=CLASSES)

# heatmap에서 중요한 것은 대각선의 값이고, 그 외의 값은 모두 오차- 잘못 예측한 것
# 해당 맵을 토대로 모델의 어떤 부분을 수정할 지 결정할 수 있음!
