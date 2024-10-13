Regression에서 발생하는 것
Underfitting
모델의 예측값과 실제 값이 크게 차이가 나는 것 - 제대로 훈련이 덜 된 것
Overfitting
모델의 예측값이 실제 값 대부분과 오류 거의 없이 일치 - 훈련이 과도하게 진행된 것
훈련한 데이터 말고 새로운 데이터에 잘 맞지 않는 문제가 발생함


잘 맞는 것은 모델의 예측값과 실제 값들이 오차가 크지 않으면서 들어맞는 것

Bias, Variance

Bias - 실제 값과 예측값의 차이가 발생할 때 이를 보정하기 위한 값

Bias가 높다는 것은 모델의 기본 모델이 잘 들어맞지 않는다는 것을 뜻하기도 함

Variance
예측된 훈련용 데이터와 예측된 평가된 데이터의 차이

parameter가 늘어나면 모델의 복잡성이 증가
그렇게 되면 bias는 줄고 variance가 증가

높은 bias
학습이 빠르게 진행되고, parameter가 적어 이해하기 쉬우나
복잡한 문제에 성능이 (예측의 정확성) 떨어짐 (Underfitting)

높은 Variance
서로 다른 훈련 데이터가 사용되면 예측 값이 크게 변화함
Low variance algorithms : Linear Regression, LDA, Logistic Regression
High variance algorithms: Decision Trees, kNN, SVM
High Variance - 더 많은 파라미터를 가진 것
훈련 성능은 좋고, 평가 성능은 좋지 않음 - 일반화 하기 어려워짐
overfitting이 발생하기 쉬움

Bias, Variance Tradeoff

목표
low bias/ low variance
예측 성능이 좋을 것(계산 시간도 적고 예측된 정확성도 높을 것
bias, variance는 서로 다른 방향으로 움직임
선형의 ML 알고리즘은 대체로 높은 bias, 낮은 variance를 가짐
비선형의 ML 알고리즘은 대체로 낮은 bias, 높은 variance를 가짐
비선형 ML 알고리즘은 tuning을 위한 별도의 파라미터(hyperparameter)를 갖는 경우가 있음
