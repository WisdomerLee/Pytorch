# 해당 코드의 기능을 쓰기 위한 data는 train/cat, train/dog, test/cat, test/dog 폴더에 각각 개, 고양이 그림이 들어가 있음

from collections import OrderedDict
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torchvision
from torchvision import transforms, models # models는 이미 훈련된 모델을 가져올 때 쓰이는 library
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score



# 원본 데이터는 다음의 링크에서 가져옴 https://www.microsoft.com/en-us/download/datasets

train_dir = 'train'
test_dir = 'test'

transform = transforms.Compose([
  transforms.Resize(255),
  transforms.CenterCrop(224),
  transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

dataset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

def imshow(image_torch):
  image_torch = image_torch.numpy().transpose((1, 2, 0))
  plt.figure()
  plt.imshow(image_torch)

X_train, y_train = next(iter(train_loader))

image_grid = torchvision.utils.make_grid(X_train[:16, :, :, :], scale_each=True, nrow=4)

imshow(image_grid)

# 이미 훈련된 모델 받아서 가져오기
model = models.densenet121(pretrained=True) # 이와 같이 모델을 설정할 때 이미 훈련된 파라미터를 같이 가져올 수 있음!!

# 훈련된 모델의 layer들의 파라미터 업데이트 막기 - Freeze
for params in model.parameters():
  params.requires_grad = False

print(model.classifier) # 해당 내용으로 classifier의 in_features를 확인할 것 - 해당 내용은 그대로 classifier를 바꿔치기 할 때 in_features에 활용

# 모델의 분류 layer를 바꿔치기 하기
model.classifier = nn.Sequential(OrderedDict([
  ('fc1', nn.Linear(1024, 1)),
  
]))


opt = optim.Adam(model.classifier.parameters())
loss_function = nn.BCELoss()
train_losses = []

NUM_EPOCHS=10
for epoch in range(NUM_EPOCHS):
  train_loss=0
  test_loss=0
  for bat, (img, label) in enumerate(train_loader):
    
    opt.zero_grad()
    
    output = model(img)
    
    loss = loss_function(output.squeeze(), label.float())
    
    loss.backward()
    
    opt.step()

    train_loss += loss.item()
    
  train_losses.append(train_loss)
  print(f"epoch: {epoch}, train_loss: {train_loss}")

sns.lineplot(x=range(len(train_losses)), y=train_losses)

fig = plt.figure(figsize=(10, 10))
class_labels = {0:'cat', 1:'dog'}
X_test, y_test = iter(test_loader).next()
with torch.no_grad():
  y_pred = model(X_test)
  y_pred = torch.argmax(y_pred, dim=1)
  y_pred = [p.item() for p in y_pred]

for num, sample in enumerate(X_test):
  plt.subplot(4,6,num+1)
  plt.title(class_labels[y_pred[num]])
  plt.axis('off')
  sample = sample.cpu().numpy()
  plt.imshow(np.transpose(sample, (1, 2, 0)))

acc = accuracy_score(y_test, y_pred)
print(f"Accurary score: {np.round(acc * 100, 2)} %")
