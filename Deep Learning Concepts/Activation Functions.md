
Rectified Linear Unit (ReLu)
Phi = Max(0, x)
가장 많이 쓰이고,
Non-Linear

Leaky Rectified Linear Unit(Leaky ReLu)
Phi(x) =  x(if x>0), alpha*x (else)
alpha는 통상적으로 0.01이 많이 쓰임
negative input에 0 대신 아주 작은 기울기를 가진 값이 있음
기울기는 0가 아님 - 0인 순간 ReLu가 됨


Hyperbolic Tangent(tanh)
$$Phi(x) = (e^x-e^-x)/(e^x+e^-x)$$

Non-Linear
좁은 영역을 제외하고는 상대적으로 평평함
기울기도 좁은 영역을 제외하고 작음
vanishing gradient problem을 마주할 가능성이 있음

Sigmoid
$Phi(x) = 1/(1+e^-x)$
좁은 영역을 제외하고는 상대적으로 평평함
기울기도 좁은 영역을 제외하고 작음
vanishing gradient problem을 마주할 가능성이 있음
결과값은 0과 1사이에 존재

Softmax
multi-class prediction에 활용
$Phi(x) = e^x_i \over \sum _i e^x_i $

각 확률로 계산되는데, 그 확률의 총 합은 반드시 1이 되어야 함
