import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

os.getcwd()

# transform, 불러오기
transform = transforms.Compose([
  transforms.Resize(32),
  transforms.Grayscale(num_output_channels=1),
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

batch_size = 4
trainset = torchvision.datasets.ImageFolder(root='data/train', transform=transform)
testset = torchvision.datasets.ImageFolder(root='data/test', transform=transform)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

# 중요한 사항 - 아래와 같은 폴더 구조를 갖지 않으면 훈련이 진행되지 않음
# 훈련용 데이터는 train, test 폴더에 나뉘어있어야 하고
# train, test 폴더에는 그림이 어떤 클래스에 있는지 폴더별로 구분되어 있어야 함


def imshow(img):
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))
  plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images, nrow=2))

class ImageClassificationNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 3) # 출력 Batchsize, 6, 30, 30
    self.pool = nn.MaxPool2d(2,2) # 출력 Batchsize, 6, 15, 15
    self.conv2 = nn.Conv2d(6, 16, 3) # 출력 Batchsize, 16, 13, 13
    self.fc1 = nn.Linear(16*6*6, 128) # 들어갈 입력의 크기는 어떻게 계산? - images[0].shape로 데이터의 모습을 확인하고, 다음 pool에서는 Batchsize, 16, 6,6의 크기로 출력되어야 함
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 1)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.conv1(x) # 출력 Batchsize, 6, 30, 30
    x = F.relu(x)
    x = self.pool(x) # 출력 Batchsize, 6, 15, 15
    x = self.conv2(x) # 출력 Batchsize, 16, 13, 13
    x = F.relu(x)
    x = self.pool(x) # 출력 Batchsize, 16, 6, 6
    x = torch.flatten(x, 1) # x를 1차원으로 펼치기! 출력 - Batchsize, 16*6*6
    x = self.fc1(x) # 출력: BS, 128
    x = self.relu(x) 
    x = self.fc2(x) # 출력: BS, 64
    x = self.relu(x)
    x = self.fc3(x) # 출력: Batchsize, 1
    x = self.sigmoid(x)
    return x

# 모델 초기화
model = ImageClassificationNet()

# loss, optimizer
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8) # 여기서 쓰인 것은 stochastic gradient descent 함수

# 훈련
NUM_EPOCHS= 10
for epoch in range(NUM_EPOCHS):
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    # optimizer의 gradient 초기화
    optimizer.zero_grad()
    # 모델 에측
    outputs = model(inputs)
    # loss 계산
    loss = loss_fn(outputs, labels.reshape(-1, 1).float()) # 데이터 타입이 달라서 문제가 생긴다면 outputs.shape, labels.shape등으로 데이터의 타입을 확인, 일치하는지 반드시 확인할 것

    # loss 역전파 - gradient 적용
    loss.backward()
    # weights update
    optimizer.step()

    if i % 100 == 0:
      print(f'Epoch {epoch}/{NUM_EPOCHS}, Step{i+1}/{len(train_loader)}', f'Loss: {loss.item():.4f}')

y_test = []
y_test_pred = []
for i, data in enumerate(test_loader):
  inputs, y_test_temp = data
  
