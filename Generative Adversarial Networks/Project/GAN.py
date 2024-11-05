
import torch
from torch.utils.data import DataLoader
from torch import nn

import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import seaborn as sns

sns.set(rc={'figure.figsize': (12,12)})

# 훈련 데이터 만들기
TRAIN_DATA_COUNT=1024

theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)])

x = 16 * (np.sin(theta) ** 3)
y = 13 * np.cos(theta)- 5 * np.cos(2*theta) - 2 * np.cos(3* theta) - np.cos(4*theta)

sns.scatterplot(x=x, y=y)

train_data = torch.Tensor(np.stack((x,y), axis=1))
train_labels = torch.zeros(TRAIN_DATA_COUNT)
train_set = [
  (train_data[i], train_labels[i]) for i in range(TRAIN_DATA_COUNT)
]

# dataloader
BATCH_SIZE = 64
train_loader = DataLoader(
  train_set, batch_size=BATCH_SIZE, shuffle=True
)

# discriminator, generator 초기화!
discriminator = nn.Sequential(
  nn.Linear(2, 256),
  nn.ReLU(),
  nn.Dropout(0.3), # 이것은 neuron간 연결 중에 무작위로 끊어지는 비율 - fully connected가 아닌 임의로 연결되는 부분, 그렇지 않은 부분들이 나뉘어 패턴을 배우거나 할 때 weight의 업데이트 쪽에 큰 영향을 주게 됨
  nn.Linear(256, 128),
  nn.ReLU(),
  nn.Dropout(0.3),
  nn.Linear(128, 64),
  nn.ReLU(),
  nn.Dropout(0.3),
  nn.Linear(64, 1),
  nn.Sigmoid()
)

generator = nn.Sequential(
  nn.Linear(2, 16),
  nn.ReLU(),
  nn.Linear(16, 64),
  nn.ReLU(),
  nn.Linear(64, 2),
)

# 훈련
LR = 0.001
NUM_EPOCHS = 200
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters())
optimizer_generator = torch.optim.Adam(generator.parameters())

for epoch in range(NUM_EPOCHS):
  for n, (real_samples, _) in enumerate(train_loader):
    # discriminator를 훈련시킬 데이터
    real_sample_labels = torch.ones((BATCH_SIZE, 1))
    latent_space_samples = torch.randn((BATCH_SIZE, 2))
    generated_samples = generator(latent_space_samples)
    generated_samples_labels = torch.zeros((BATCH_SIZE, 1))
    all_samples = torch.cat((real_samples, generated_samples), dim=0)
    all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels), dim=0)
    # 실제 데이터와 가짜 데이터를 준비하고, 하나로 합쳐 discriminator로 전달!
    # generator와 discriminator가 같은 비율로 훈련해야 하기 때문에 epoch를 2로 나누어 아래와 같이 훈련하는 경우가 대부분!!!
    
    if epoch % 2 == 0:
      # discriminator 훈련
      discriminator.zero_grad()
      output_discriminator = discriminator(all_samples) # 합쳐진 sample이 discriminator로 전달
      loss_discriminator = loss_function(output_discriminator, all_samples_labels) # discriminator가 구분한 것과 실제 데이터를 비교하여 loss 계산
      loss_discriminator.backward()
      optimizer_discriminator.step()
      
    if epoch % 2 == 1:
      # generator를 훈련시킬 데이터
      latent_space_samples = torch.randn((BATCH_SIZE, 2)) # 이것은 generator에 전달될 기본 그림을 만들 때 초반으로 들어갈 잡티(noise)

      # generator 훈련
      generator.zero_grad()
      generated_samples = generator(latent_space_samples) # 임의의 것을 기반으로 샘플 생성
      output_discriminator_generated = discriminator(generated_samples) # discriminator가 sample을 판별
      loss_generator = loss_function(output_discriminator_generated, real_samples_labels) # discriminator가 구분한 것과 실제 데이터와 비교하여 loss 계산
      loss_generator.backward()
      optimizer_generator.step()
      

      if epoch % 10 == 0:
        print(epoch)
        print(f"Epoch {epoch}, Discriminator Loss {loss_discriminator}")
        print(f"Epoch {epoch}, Generator Loss {loss_generator}")
        with torch.no_grad():
          latent_space_samples = torch.randn(1000, 2)
          generated_samples = generator(latent_space_samples).detach()
        plt.figure()
        plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
        plt.xlim((-20, 20))
        plt.ylim((-20, 15))
        plt.text(10, 15, f"Epoch {epoch}")
        plt.savefig(f"train_progress/image{str(epoch).zfill(3)}.jpg")

# 결과 확인
latent_space_samples = torch.randn(10000, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:0], generated_samples[:, 1], ".")
plt.text(10, 15 f"Epoch {epoch}")
plt.show()
