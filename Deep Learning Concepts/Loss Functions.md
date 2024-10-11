
훈련 도중에 모델을 평가하는 과정이 들어가는데
점진적으로 개선해야 하고, 그 이유는 최적화 때문
훈련 과정을 통해 해당 값을 통해 나오는 값을 최소로 줄이게 됨
하나의 모델에 여러 loss function을 가질 수 있음 - 각 output variable에 해당되는 값들에 loss function이 하나씩 붙을 수도 있음

대체로 Regression, Classification을 위한 함수들이 많은 편

# Regression Losses
Mean squared Error MSE = $\sum ((i=1,n) (y_i-y^^_i)^2 \over n)$
Mean Absolute Error MAE = $\sum ((i=1, n) |y_i-y^^_i| \over n) $
Mean Bias Error MBE = $\sum (i=1, n) (y_i-y\hat_i) \over n$
Output layer - 1 node
Typical activation function: linear

# Binary Cross Entropy
binary classification에 적용
매우 흔히 쓰임
Output layer - 1 node
Typical activation function : sigmoid

$CE = -(y_i logy_i)+(1-y_i)log(1-y_i)$

# Hinge Loss
SVM loss로도 부름
binary classification에 적용
maximum margin classifier을 사용할 때 많이 쓰임, 두 가지를 분류할 때 데이터에서 가장 멀리 떨어진 구분 선을 긋도록 할 때
Output layer - 1 node
Typical activation function : sigmoid
$HingeLoss = \sum j!=yi max(0, s_i-s_y_i+1)$

# Multi-Label Cross Entropy
multi-label classification에 가장 널리 쓰임
output - n nodes n - label의 갯수
Typical activation function : softmax

