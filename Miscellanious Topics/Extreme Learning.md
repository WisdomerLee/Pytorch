# Introduction
일반적인 Neural Networks
훈련에 오랜 시간이 걸리고, 예측이 빠름

발생할 수 있는 문제
Network가 만약 데이터의 변화로 주기적으로 재훈련이 필요한 상태라면... 긴 훈련 시간 때문에 불가능할 수도 있음

# Extreme Learning의 장단
Extreme Learning은 짧은 시간에 훈련이 가능하나, 일반 Neural Network에 비해 성능이 낮아지는 문제가 있음

# ELM 이론
ELM은 하나의 숨겨진 layer로 feedforward neural network - 1개의 hidden layer를 가져 연산이 매우매우 빠른 편
훈련이 빠름
기본적인 NN과 비슷, 학습과정이 없음 (backpropagation이 없음)

1. hidden layer의 neuron에 파라미터들이 무작위로 지정
2. hidden layer를 거쳐 입력 계산
3. output의 weight를 계산

hidden layer가 단층이기 때문에 정확도는 최근의 모델 구조보다 떨어짐
그러나 학습에 걸리는 시간이 파격적으로 줄어들기 때문에 재훈련을 자주 시켜야 할 경우 선택할 수 있음
