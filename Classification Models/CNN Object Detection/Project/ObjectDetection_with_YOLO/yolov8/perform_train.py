
from ultralytics import YOLO

# sources:
# https://docs.ultralytics.com/cli/
# https://docs.ultralytics.com/cfg/

model = YOLO("yolov8n.yaml") # model을 새로 만들고

model = YOLO("yolov8n.pt") # 기존의 훈련된 파라미터를 불러오기

# 모델 훈련
results = model.train(data='train_custom/masks.yaml', epochs=1, imgsz=512, batch=4, verbose=True, device='cpu')
# gpu가 없는 상황을 가정하고 cpu로 훈련, gpu로 훈련할 때보다 매우매우 느림 주의
# 훈련된 모델 추출
model.export()

import torch
# 훈련용 GPU 사용 가능인지 확인하기
torch.cuda.is_available()

