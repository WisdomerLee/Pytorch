# Inception

# Recap
Resnet은 보다 적은 파라미터로 skip connections라는 기법으로 성능을 높게 만들었음

Inception은 Resnet의 구조를 기반으로 만들어짐

# Introduction
구글에서 개발
GoogleNet(v1) - 2014년에 개발
v4는 2016년에 개발
여러 inception modules들로 구성되어 있음

평행 접근 방식 - 아래에 설명하는 방식은 GoogleNetv1의 방식
MaxPool 3x3 -> Conv 1x1 -> BatchNorm -> ReLU 
Conv 1x1 -> BatchNorm -> ReLU -> Conv 3x3 -> BatchNorm -> ReLU
Conv 1x1 -> BatchNorm -> ReLU -> Conv 5x5 -> BatchNorm -> ReLU
Conv 1x1 -> BatchNorm -> ReLU
위와 같이 평행하게 통과하는 층이 다르게 설정하고
그 결과물을 하나로 통합

# 핵심 특징
inception modules: multiple branches, 필터 크기 다름
feature extraction: filters with different size -> 다른 크기의 특성을 확인할 수 있음
dimensional reduction: feature maps의 차원을 "bottlenect layers"로 줄일 수 있음
axuilary classifiers

평행하게 진행되는 접근을 어떻게 구현할 수 있는가?
