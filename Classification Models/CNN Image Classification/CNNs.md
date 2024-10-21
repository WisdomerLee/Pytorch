# Convolutional Neural Networks

Deep learning network의 한 종류
대부분의 computer vision과 관련된 deep learning

CNN은 local pattern을 배움
local patterns translational invariant
layer들은 서로 다른 구조적인 패턴을 배움 - 어느쪽 조합은 턱, 어느 쪽 조합은 눈, 등

# Convoluation
Input Image Matrix  Convolutional Filter(Edge Detector, Blur 등등의 여러 필터 있음) -> Feature map으로 변환
입력된 그림 그대로 전달할 수도 있으나, 그러면 연산에 필요한 리소스가 지나치게 많아, 그림의 정보를 압축할 필요가 있는데, 그 때 Convolution이 사용됨
그림의 정보를 적당히 압축하여 넘길 때 Convolutional Filter로는 주로 3x3 matrix가 많이 쓰이고, 5x5도 드물게 쓰이기도 함

# Convolution - Stride
Stride = step size
Stride = n
feature를 추출할 때 pixel을 한 단위 옆이 아닌 n 단위 옆으로 이동
압축되는 정도, 손실되는 정도를 stride로 조절할 수 있음

# Convoluational Layer
Input Image Matrix - Convolutional Filter -> Convolutional Layer (Feature Maps)

# Max Pooling
Feature Map으로 줄인 그림의 정보를 더 압축하는 과정으로, Feature Map의 이웃들 중에서 가장 높은 값을 대표값으로 추출하여 데이터의 차원을 줄이는 것

# Network Setup
input layer에서
feature map size는 input layer로 들어갈 matrix의 세로 길이를, featuremap count는 matrix의 가로 길이를 결정

Input layer - Convolutional Layer - Max Pooling - Convolutional Layer - Max Pooling - Dense Layer - Output Layer Softmax
일반적으로 위와 같은 구조를 기본 구조로 갖고 있음

# Network를 통과할 때의 변화
feature map의 갯수는 layer를 통과하면서 증가하고, feature map의 크기는 줄어들게 됨

# 장점, 단점
Computer vision과 관련된 임무를 수행할 때 매우 강력한 도구
매우 높은 예측 품질을 보장

파라미터가 많고, 훈련이 많이 필요하고, 연산 비용이 매우 비싼 편

