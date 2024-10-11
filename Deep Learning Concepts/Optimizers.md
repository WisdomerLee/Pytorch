훈련이 진행될 때 model의 weight를 loss 함수의 값을 최소화하도록 업데이트하는데
이를 담당하는 것이 Optimizer

Loss function의 값을 기반으로 weights들을 얼만큼 보정해야 할 지 계산
Brute force - 모든 combination을 확인 - 매우 나쁨, 가장 최적의 경우를 계산하지만 계산되는 시간이 매우 매우 오래 걸림
Educated trial and error - 시행착오로 줄이는 방식


# Gradient Descent
Loss function의 gradient를 계산

weights 초기화
weights의 작은 변화를 계산
각각 weights에 적용
minimum을 찾을 때까지 반복

해당 방식의 문제
local minima에 갇힐 수도 있음

이것을 극복할 수 있는 방식은...?
Convex loss function
Learning rate 조절

Learning rate는 weight의 변화 값을 얼만큼씩 조정할 수 있는지를 결정
높은 Learning rate는 큰 폭으로 값을 변화하나, minimum을 건너 뛸 수 있는 부작용이 있음
낮은 Learning rate는 정확하게 추적할 수 있으나, 시간이 매우 오래 걸리는 문제가 있음

다른 방식
# Adagrad
learning rate를 특징으로 받아들여서 learning rate = f(weights)로 계산할 수 있음
dataset을 분할할 때 잘 동작함
Learning rate는 시간이 지나면서 감소하며, 종종 매우 작아질 때도 있음
Adaprop, RMSprop 등의 방식으로 해당 문제를 극복

# Adam
Adaptive momentum estimation
momentum을 적용 - 기존의 gradient를 현재의 gradient를 계산에 포함 
Widespread

# 기타
Stochastic Gradient Descent, Batch gradient descent
