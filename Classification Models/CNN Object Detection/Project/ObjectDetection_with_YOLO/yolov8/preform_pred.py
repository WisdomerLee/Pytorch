
from ultralytics import YOLO

model = YOLO('yolov8n.pt') # 모델 불러오기 - 역시 pt로 저장된 이미 학습된 모델을 불러오기

result = model('test/kiki.jpg')

result

# 만약 커맨드 라인처럼 실행하고 싶다면 아래와 같이
# 일반 YOLO
!yolo detect predict model=yolov8n.pt
source = 'test/kiki.jpg' conf=0.3

# Mask 테스트하기!
!yolo detect predict model=train_custom/
masks.pt source = 'train_custom/test/images/IMG_0742.MOV' conf=0.3

