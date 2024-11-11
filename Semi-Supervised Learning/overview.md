# Semi-Supervised Learning??
인공지능 학습은 Supervised, Unsupervised, Reinforcement Learning으로 나뉘고
연속적, 분류된 데이터를 학습하느냐에 따라 그 형태가 달라지는데

semi-supervised learning은 그 경계에 있음

# 문제
데이터를 훈련시키려고 보면 모든 데이터에 label을 붙여야 하는데, 
일부에만 붙어있고, 대부분이 label이 없을 경우 > 훈련을 시켜야 할 때 어떻게 할까?

labeled된 데이터만 훈련에 활용하기! > 매우 간단하나, 그럼 데이터의 숫자가 줄어들게 됨...
나머지 데이터에 label을 붙이고 모든 dataset을 훈련시키기 > 사람의 시간과 비용이 많이 들어감...

Semi-supervised model로 훈련하기!를 하면 이 문제를 해결하는 또다른 방법이 될 수 있음

# Paper

https://arxiv.org/pdf/1803.07728.pdf

Unsupervised Representation Learning by Predicting Image Rotations
Spyros 

그림의 회전을 알고리즘으로 이해시키기를 시도

# Result

Exploring Self-Supervised Regularization for Supervised and Semi-Supervised Learning
Flyreel AI Research
https://arxiv.org/pdf/1906.10343.pdf

CIFAR-100으로 분류할 경우
데이터는 label이 붙은 것과 붙지 않은 데이터가 섞여있음!
오히려 supervised learning보다 semi supervised learning이 더 나은 결과를 보이기도 함!

# 활용할 Dataset
Panda / Bear Image Classification

https://www.kaggle.com/datasets/mattop/panda-or-bear-image-classification

# Self-supervised Task
Transformation - Roation

# SESEMI Architecture
self-supervised task를 회전 예측에 사용

Supervised - 분류 labeled된 데이터
Self-Supervised - 랜덤으로 회전된 데이터!
Supervised Cross Entropy
Self Supervised Cross Entropy
Semi - Supervised Loss

