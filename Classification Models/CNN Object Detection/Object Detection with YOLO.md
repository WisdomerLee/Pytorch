# 사물 인식 알고리즘
사물 인식 알고리즘은 크게 하나(One Stage), 혹은 두 단계(Two Stage)의 알고리즘을 가짐

Two Stage에 쓰이는 모델들은 RCNN, Fast RCNN, Faster RCNN, RFCN, Mask RCNN
One Stage에 쓰이는 모델은 YOLO, SSD


# YOLO?
You only look once
2016년에 개발
one-stage algorithm
24개의 CNN layer와 2개의 fully connected layers로 구성

YOLO는 사물 인식 알고리즘 중에 간단한 편에 속함

# 동작 원리는?
one-stage: 객체가 있을 영역을 네모로 둘러싸고, 동시에 그것이 무엇인지를 분류하는 과정을 같이 진행
그림을 grid(잘게 자른 네모)로 작게 쪼개기
나뉘어진 grid마다 객체로 인식되는 부분을 네모로 둘러싸고, 오브젝트와 겹치는 구간이 특정 이상이 되는 것을 골라내기
각 grid는 어느 사물에 해당하는지 확률을 계산
box confiedences * class probabilites

최종적으로 네모로 둘러싼 영역과 grid에서 어느 사물로 인식한 부분을 같이 합쳐 해당 네모 사물이 어느 사물인지를 구분

# YOLO v7은 이전의 YOLO와 어떤 부분이 달라졌는가?
YOLOv7은 network 구성이 달라짐
Extended Efficient Layer aggregation - 확장된, 효율적인 특성 추출 layer들로 구성
layer들이 기억을 갖고 있고, back-propage에서 gradient를 계산하는 방식이 조금 더 차이가 남 (역전파의 가장 큰 문제인 뒤로 갈 수록 해당 요소의 영향력이 줄어드는 문제를 해결하기 위해)
Model Scaling
네트워크 깊이, 너비가 확장되고, layer는 조금 더 축약 됨

# 한계
Fast RCNN에 비하면 빠르나, 정확도가 낮음
작은 사물들은 하나하나 구분하지 못하고 커다랗게 뭉뚱그려 하나의 그룹으로 파악함

