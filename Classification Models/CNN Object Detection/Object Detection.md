# 사물 감지
사물 감지는 그림에서 오브젝트가 어느 위치에 있는지 감지하는 것이고
사물이 있을 것으로 추정되는 위치를 네모로 표시

매우 다양한 알고리즘이 있음
Fast R-CNN
Faster R-CNN
YOLO(You only look once)
SSD(Single Shot Detector)

일반적으로 컴퓨터 자원을 많이 소모하는 훈련이기도 함
모델 설정하는 것도 번거롭고

# detecto
detecto라는 파이썬 패키지가 있음
PyTorch 기반이고
컴퓨터 비전 관련, 사물 감지 관련된 모델을 5줄로 표현할 수 있다고 하는 라이브러리
Faster R-CNN, ResNet-50, FPN 등의 모델을 기반으로 동작


# coding
여기서 활용할 데이터는
사과, 바나나가 있는 사물 객체 사진 데이터
