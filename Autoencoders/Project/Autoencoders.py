
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils

path_images = 'data/train'

transform = transforms.Compose([
  transforms.Resize((64,64)),
  transforms.Grayscale(num_output_channels=1),
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

dataset = ImageFolder(root=path_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

LATENT_DIMS = 128

# Encoder class를 정의
class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 3) # out : BS, 6, 62, 62
    self.conv2 = nn.Conv2d(6, 16, 3) # out : BS, 16, 60, 60
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(16*60*60, LATENT_DIMS)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x


# Decoder class를 정의
class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(LATENT_DIMS, 16*60*60) # Decoder는 encoder의 역으로 진행되는 것을 고려하면 encoder와 입력, 출력에 들어가는 숫자가 반대가 됨
    self.conv2 = nn.ConvTranspose2d(16, 6, 3) # out : BS, 16, 60, 60 # Conv2d가 아닌, Transpose된 2d를 이용해야 함 -> 행렬에서 역으로 곱하는 것을 생각해보자!
    self.conv1 = nn.ConvTranspose2d(6, 1, 3) # out : BS, 6, 62, 62 
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
  def forward(self, x):
    x = self.fc(x)
    x = x.view(-1, 16, 60, 60) # 2d (bs, feature) -> 4d
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv1(x)
    x = self.relu(x)
    return x

# AutoEncoder class를 정의
class Autoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

# 데이터 형태 확인
# input = torch.rand((1, 1, 64, 64))
# model = Autoencoder()
# model(input).shape

model = Autoencoder()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCHS = 30

for epoch in range(NUM_EPOCHS):
  losses_epoch = []
  for batch_idx, (data, target) in enumerate(dataloader):
    data = data.view(-1, 1, 64, 64)
    output = model(data)
    optimizer.zero_grad()
    loss = F.mse_loss(output, data)
    losses_epoch.append(loss.item())
    loss.backward()
    optimizer.step()
  print(f"Epoch: {epoch} \tLoss: {np.mean(losses_epoch)}")

def show_image(img):
  img = 0.5 * (img + 1) # denormalize
  npimg =  img.numpy()
  plt.imshow(np.transpose(npimg, (1,2,0)))

print('original')
plt.rcParams["figure.figsize"] = (20,3)
show_image(torchvision.utils.make_grid(images))

print('latent space')

latent_img = model.encoder(images)
latent_img = latent_img.view(-1, 1, 8, 16)
show_image(torchvision.utils.make_grid(latent_img))

print('reconstructed')
show_image(torchvision.utils.make_grid(model(images)))

image_size = images.shape[2] * image.shape[3] * 1
compression_rate = (1-LATENT_DIMS/image_size) * 100
compression_rate

  
    
