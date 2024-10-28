# YOLOv7을 이용할 프로젝트
Face Mask Detection
Data source: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
classes: with_mask, without_mask, mask_weared_incorrect
853개의 이미지
label은 Pascal Voc format으로 되어있음

YOLO Project를 복사
Get Weights
Config file 적용
data file 적용
Raw Images, Labels 얻기
Image, Label의 데이터 전처리 과정
훈련하기
모델 테스트

# YOLO project 클론 복사하기
 clone YOLO Project : git clone https://github.com/WongKinYiu/yolov7.git
 weights 얻기 - download https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
 yolo7 폴더에 저장
 config file 적용 - yolov7/cfg/training/yolov7-e6e.yaml
 해당 파일에서 nc: 7을 3으로 수정할 것
 
# Data file 적용하기
yolov7/data/masks.yaml
```
train: ./train
val: ./val
test: ./test

# Classes
nc: 3
names: ['with_mask', 'without_mask', 'mask_weared_incorrect']

```

# Image 파일, Labels 얻기
https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

# Data Preparation Images and Labels
Pascal Voc Labels를 Yolo 포맷으로 변경
data를 subfolder로 나누기

project folder
  annotations
  images
  yolo7

Project folder
  yolo7
    val
      images
      labels
    test
      images
      labels
    train
      images
      labels

# 훈련하기
python train.py --weights yolov7-e6e.pt --data "data/masks.yaml" --workers 1 --batch-size 4 --img 640 --cfg cfg/training/yolov7-masks.yaml --name yolov7 --epochs 50
으로 설정이 매우 많음

# 모델 테스트
python detect.py --weights runs/train/yolov73/weights/best.pt --conf 0.4 --img-size 640 --source ./test/images/file_to_test.png

