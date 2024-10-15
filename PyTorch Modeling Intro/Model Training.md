PyTorch Functional API : https://jeancochrane.com/blog/pytorch-functional-api

어느 프로세스의 어느 단계에서 어느 객체가 어떻게 영향을 받는지를 반드시 이해하고 넘어갈 것
위의 링크는 좋은 참고가 될 것

Model, Optimizer, Loss Function, Gradient Update, loss.backward(), Model Parameter, optimizer.step()


# State(상태)와 object(객체)
object와 상태에 영향을 주는 것
## Model
torch.nn을 기반으로 layer, activation functions등을 갖고 있음
## Optimizer
Model parameter는 optimizer에 공유되어야 함
## Loss Function
loss 계산

영향을 받는 것
Model parameters
Gradients

gradients가 0에 가까운 값을 갖도록 Model parameter가 변화

# Training Loop
일반적인 훈련 구조는 아래와 같고 기본 코드 구조도 아래와 같은 순서로 진행됨을 기억하자
```
for epoch in range(number_epochs):
  for j, data in enumerate(train_loader):
    # optimizer - 초기화 (기존의 값을 토대로 새 값을 계산할 준비)
    optimizer.zero_grad()
    # forward pass - 모델이 입력을 토대로 예측값 계산
    y_hat = model(data[0])
    # loss 계산 - 모델이 예측한 값과 실제 데이터의 차이 계산
    loss = loss_fun(y_hat, data[1])
    losses.append(loss.item())

    # backpropagation - loss를 토대로 gradient를 계산 - 파라미터들이 어느 정도로 바뀌어야 하는지 계산되는 기준
    loss.backward()

    # 모델 파라미터 업데이트 - loss의 backward에서 계산된 내용을 토대로 optimizer가 모델의 파라미터를 업데이트
    optimizer.step()
```

# Clear gradients
optimizer는 gradient를 계산 촉진
각각 새로 계산될 forward, backward gradient는 지워져야 함 
```
optimizer.zero_grad()
```
optimizer는 gradient를 갖고 있는가? 모델이 아니라?

# Forward pass
모델이 예측하는 값을 계산하는 과정
모델의 현재 파라미터를 토대로 값을 계산
```
y_hat = model(data[0])
```

# Loss calculation
loss 값 계산
예측과 실제 데이터를 이용하여 오차가 얼마나 났는지 계산
```
loss = loss_fun(y_hat, data[1])
```

# Gradient calculation
loss 함수는 모든 node(모델 파라미터)들의 모든 gradient를 계산
각 파라미터들이 loss함수의 편미분 값을 들고 있게 됨
gradients의 값 자체도 변화
model layer들이 사용되고, model들의 tensor gradients가 update됨
```
loss.backward()
```

# Weight update
Gradients 가 알려지면, weights는 그에 맞게 바뀌어야 함
optimizer의 step()함수가 이 기능을 수행
model parameter가 이때 업데이트 됨 - 모델 자체는 불러와지지 않고, 모델의 파라미터들만 영향
```
optimizer.step()
```

# Pytorch의 모델 훈련 방식을 사용할 때의 장단점
장점은 Deep Learning을 활용하는데 매우 수월해지는것
단점은 Model, loss, optimizer가 파라미터를 수정하는데, 내부 상황을 정확히 알 수 없게 된다는 부분도 존재함

