
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
from sklearn.metrics import accuracy_score
from PIL import Image
import seaborn as sns

BATCH_SIZE = 10
DEVICE = torch.device("cuda:0" if torch.cuda.isavailable() else "cpu")
NUM_EPOCHS = 50
LOSS_FACTOR_SELFSUPERVISED=0

transform_super = transforms.Compose([
  transforms.Resize(32),
  transforms.Grayscale(num_output_channels=1),
  transforms.ToTensor(),
  transforms.Normalize((0.5, ), (0.5, ))
])

# label이 없는 Dataset과 DataLoader를 구현하기!!
class UnlabledDataset(Dataset):
  sesemi_transformations = {0:0, 1:90, 2:180, 3:270}
  def __init__(self, folder_path):
    super().__init__()
    self.folder_path = folder_path
    self.image_names = os.listdir(folder_path)
    self.images_full_path_names = [f"{folder_path}/{i}" for i in self.image_names]
    
  def __len__(self):
    return len(self.image_names)
  def __getitem__(self, idx):
    img = Image.open(self.images_full_path_names[idx])
    # random sesemi transformations
    transformation_class_label = random.randint(0, len(self.sesemi_transformations))
    # 랜덤으로 선택된 변환을 적용하기
    angle = self.sesemi_transformations[transformation_class_label]
    data = transform_super(img)
    data = transforms.function.rotate(img=data, angle=angle)
    return data, transformation_class_label

# label이 없는 DataSet 지정하기 - UnlabeledDataset 클래스를 이용할 것
folder_path = 'data/unlabeled'
unlabeled_ds = UnlabeledDataset(folder_path)



train_ds = torchvision.datasets.ImageFolder(root='data/train', transform=transform_super)
test_ds = torchvision.datasets.ImageFolder(root='data/test', transform=transform_super)

# supervised dataloader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# unsupervised dataloader 지정하기
unlabeled_loader DataLoader(unlabeled_ds, batch_size=BATCH_SIZE)

# Model 설정하기!

class SesemiNet(nn.Module):
  def __init__(self, n_super_classes, n_selfsuper_classes):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 6, 3)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(6, 16, 3)
    self.fc1 = nn.Linear(16*6*6, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc_out_super = nn.Linear(64, n_super_classes)
    self.fc_out_selfsuper = nn.Linear(64, n_selfsuper_classes)
    self.relu = nn.ReLU()
    self.output_layer_super = nn.Sigmoid()
    self.output_layer_selfsuper = nn.LogSoftmax()

  def backbone(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    return x
  # 예측을 진행할 때 둘로 나누어 진행 super, selfsuper
  
  def forward(self, x_supervised, x_selfsupervised):
    x_supervised = self.backbone(x_supervised)
    x_supervised = self.fc_out_super(x_supervised)
    x_supervised = self.output_layer_super(x_supervised)

    x_selfsupervised = self.backbone(x_selfsupervised)
    x_selfsupervised = self.fc_out_selfsuper(x_selfsupervised)
    x_selfsupervised = self.output_layer_selfsuper(x_selfsupervised)

    return x_supervised, x_selfsupervised
    


model = SesemiNet(n_super_classes=2, n_selfsuper_classes=4)
model.train()

criterion_supervised = nn.CrossEntropyLoss()
criterion_selfsupervised = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
for epoch in range(NUM_EPOCHS):
  train_loss = 0
  for i, (supervised_data) in enumerate(train_loader):
    X_super, y_super = supervised_data
    
    optimizer.zero_grad()
    y_super_pred = model(X_super)
    loss = criterion_supervised(y_super_pred, y_super)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    print(f"Epoch {epoch}: Loss {train_loss}")


sns.linepolt(x=list(range(len(train_losses))), y=train_losses)
y_test_preds = []
y_test_trues = []

with torch.no_grad():
  for (X_test, y_test) in test_loader:
    y_test_pred = model(X_test)
    y_test_pred_argmax = torch.argmax(y_test_pred, axis=1)
    y_test_preds.extend(y_test_pred_ragmax.numpy())
    y_test_trues.extend(y_test.numpy())


accuracy_score(y_pred=y_test_preds, y_true=y_test_trues)
