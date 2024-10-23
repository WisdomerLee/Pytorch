import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

os.getcwd()

# data -> 전처리
transform = transforms.Compose([
  transforms.Resize((50, 50)),
  transforms.Grayscale(num_output_channels=1),
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

# train, test dataset 설정
batch_size=4
# 폴더에 있는 데이터를 그대로 데이터로 불러옴

trainset = torchvision.datasets.ImageFolder(root='train', transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.ImageFolder(root='test', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


#
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

class ImageMulticlassClassificationNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 3) # 여기에 있는 것들도 
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6, 16, 3)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(16*11*11, 128) #(bs, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, NUM_CLASSES)
    self.relu = nn.ReLU()
    self.softmax = nn.LogSoftmax()
    

  def forward(self, x):
    x = self.conv1(x) # (bs, 6, 48, 48)
    x = self.relu(x)
    x = self.pool(x) # (bs, 6, 24, 24)
    x = self.conv2(x) # (bs, 16, 22, 22)
    x = self.relu(x)
    x = self.pool(x) # (bs, 16, 11, 11)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.softmax(x)
    return x

input = torch.rand(1, 1, 50, 50) # 이것은 model의 output shape을 확인하기 위해 임의로 넣은 입력 데이터
model = ImageMulticlassClassificationNet()
model(input).shape 
# loss, optimizer 설정
loss_fn = nn.CrossEntropyLoss() # 다중 분류 모델에서 가장 흔히 쓰이는 loss function
optimizer = torch.optim.Adam(model.parameters()) 

# 훈련
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
  print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')

y_test = []
y_test_hat = []
for i, data in enumerate(test_loader, 0):
  inputs, y_test_temp = data
  with torch.no_grad():
    y_test_hat_temp = model(inputs).round()

  y_test.extend(y_test_temp.numpy())
  y_test_hat.extend(y_test_hat_temp.numpy())

acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')

confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
