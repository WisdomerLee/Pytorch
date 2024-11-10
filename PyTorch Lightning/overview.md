# Pytorch Lightning
python package
PyTorch 기반으로 만들어진 framework
"복잡한 network를 간단한 coding으로 만들기"
매우 좋은 특징을 지님

- logging of training/ validation metrics
- creating checkpoint
- early stopping
- training on multiple GPUs, TPUs, CPUs

설치 : pip install pytorch-lightning


# 모델 만들기
model class는 다음의 4가지 함수를 갖고 있어야 함
__init__()
forward()
configure_optimizers()
training_step()

아래는 선택적으로 가질 수 있는 함수
prepare_data()
validation_step()
test_step()
predict_step()

# PyTorch와 PyTorch Lightning 비교

pytorch lightning은
import pytorch-lightning as pl
의 패키지가 추가로 필요

PyTorch
```
class LinearRegressionTorch(nn.Module):
  def __init__(self, input_size, output_size):
    super(LinearRegressionTorch, self).__init__()
    self.linear = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.linear(x)

model = LinearRegressionTorch(input_size=1, output_size=1)
model.train()

loss_fun = nn.MSELoss()

learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses = []
slop, bias = [], []
number_epochs = 1000
for epoch in range(number_epochs):
  for j, (X, y) in enumerate(train_loader):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fun(y_pred, y)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
  losses.append(float(loss.data))
```

PyTorch Lightning
```
class LitLinearRegression(pl.LightningModule):
  def __init__(self, input_size, output_size):
    super(LitLinearRegression, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    self.loss_fun = nn.MSELoss()

  def forward(self, x):
    return self.linear(x)
  def configure_optimizer(self):
    learning_rate = 0.02
    optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    return optimizer
  def training_step(self, train_batch, batch_idx):
    X, y = train_batch
    y_pred = model(x)
    loss = self.loss_fun(y_pred, y)
    self.log('train_loss', loss, prog_bar=True)
    return loss

model = LitLinearRegression(input_size=1, output_size=1)

trainer = pl.Trainer(gpus=1, precision=16, max_epochs=100)
trainer.fit(model, train_loader)

```

위가 그 둘의 예시 모델인데
1. 모델 클래스가 상속받는 대상이 달라진다 (Pytorch: nn.Module, Pytorch Lightning: pl.LightningModule)
2. optimizer가 클래스 외부에서 정의되고, 훈련 과정에서 사용되는 pytorch와 달리, Pytorch Lightning에서는 configure_optimizer라는 함수에 하나로 정의됨을 볼 수 있음
3. 훈련 루프도 달라지는데, 훈련 루프에서 loss, 등의 변수를 외부에서 선언하고 그것들을 집어넣은 상태로 훈련 과정 사이에 집어넣는 pytorch와 달리, pytorch lightning에서는 training_step 함수로 모델 훈련 방식을 결정하고, 별도로 훈련을 시켜주는 pl.Trainer를 통해 훈련이 진행됨

