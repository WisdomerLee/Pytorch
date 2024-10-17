# 모델훈련 과정은 데이터 전처리 과정과 분리되어야 함

코드를 읽기도 쉬워질 뿐더러, 모듈화를 위해서도 좋은 것

Dataset과 DataLoader가 있음
datasets를 불러오는 과정을 담당, interface 함수로 사용
사용자가 지정한 datasets에서 공통으로 쓸 수 있는 interface 함수로 사용

Dataset
 예시와 label을 포함

Dataloader
데이터를 불러오는 역할을 담당

# Custom Dataset
3개의 함수가 구현되어야 함
```
from torch.utils.data import Dataset, DataLoader
class LinearRegressionDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y
  def __len__(self):
    return len(self.X)
  def __getitem(self, idx):
    return self.X[idx], self.y[idx]
```

# DataLoader
dataloader는 dataset을 반복
dataset을 반복하면 dataset의 batches를 돌려줌
제공하는 기능은 데이터를 섞거나, sampling 방식을 지정할 수 있음
```
train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)
```
