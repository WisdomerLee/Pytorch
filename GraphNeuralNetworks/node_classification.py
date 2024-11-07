# node classification 관련된 google의 colab 쪽 내용
# https://colab.research.google.com/drive/140vFnAXggxBBvM4e8vSURUp1TaKnovzX?usp=sharing

# pytorch geometric 역시 pip install torch_geometric으로 설치 필요

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns

# dataset 불러오기
dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
print(f"Dataset: {dataset}")
print(f"# graphs: {len(dataset)}")
print(f"# features: {dataset.num_features}")
print(f"# classes: {dataset.num_classes}")

# dataset의 data가 어떻게 생겼는지는 다음과 같이 확인해볼 수 있음
data = dataset[0]
print(data)

data.edge_index
data.edge_index.shape
data.y

from collections import Counter
Counter(data.y.cpu().numpy())

# 
class GCN(torch.nn.Module):
  def __init__(self, num_hidden, num_features, num_classes):
    super().__init__()
    self.conv1 = GCNConv(num_features, num_hidden) # CNN에서는 Conv1d, Conv2d, Conv3d를 사용한 것처럼 GCN에서는 GCNConv를 사용함
    self.conv2 = GCNConv(num_hidden, num_classes)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)  
    x = x.relu() # CNN과의 차이이기도 한데, GCNConv로 나온 것은 자체적으로 relu함수를 갖고 있음
    x = F.dropout(x, p=0.2)
    x = self.conv2(x, edge_index)
    return x

# 요 근래에는 Transformer 기반의 model이 주류로 자리 잡은 편... > 최신 추세는 또 transformer에서 openai에서 발표한 system to think라는 기법으로 넘어가는 듯
# transformer는 attention이라는 기법을 활용!
# GATConv() 라는 방식으로 GCN을 만들 수 있음! < transformer 기법이 들어간 layer


model = GCN(num_hidden=16, num_features=dataset.num_features, num_classes = dataset.num_classes)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss() # 대체로 분류 모델에서 활용하는 loss 함수

# 모델 훈련
loss_lst = []
model.train()
for epoch in range(1000):
  optimizer.zero_grad()
  y_pred = model(data.x, data.edge_index)
  y_true = data.y
  loss = criterion(y_pred[data.train_mask], y_true[data.train_mask]) # CNN과의 차이... Graph Neural Network에서 활용되는데, 기본적으로 train, test로 나누지 않았음을 기억할 것 
  # data.train_mask라는 것이 있는데 별도의 tensor로, 훈련용인지, 테스트용인지가 들어있는 tensor
  loss_1st.append(loss.item())
  loss.backward()
  optimizer.step()
  print(f"Epoch: {epoch}, Loss: {loss}")
  
# train loss
sns.lineplot(x=list(range(len(loss_lst))), y=loss_lst)

# 모델 평가
model.eval()
with torch.no_grad():
  y_pred = model(data.x, data.edge_index) # 분류모델은 각 클래스에 속할 확률을 tensor로 내뱉기 때문에... 그 중에 가장 높은 확률을 가진 것으로 줄여야 함
  y_pred_cls = y_pred.argmax(dim=1)
  correct_pred = y_pred_cls[data.test_mask] == data.y[data.test_mask]
  # 맞고, 틀린 것 중에 맞는 부분만 세고, 전체 중에 얼마나 맞았는지 확인해야 함
  test_acc = int(correct_pred.sum()) / int(data.test_mask.sum())

print(f"Test Accuracy: {test_acc}")
# 결과를 그림으로 !
z = TSNE(n_components=2.fit_transform(y_pred[data.test_mask].detach().cpu().numpy()))
sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=data.y[data.test_mask])
