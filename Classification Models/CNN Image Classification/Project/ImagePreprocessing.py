import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 그림 불러오기
img = Image.open('.jpg') # 그림이 저장된 파일 경로!
img

img.size

preprocess_steps = transforms.Compose([
  transforms.Resize((300, 300)), # 모든 그림이 같은 크기의 그림이 되도록 먼저 처리, 그림의 가로, 세로 크기가 모두 같아야 함
  transforms.RandomRotation(50), # 그림을 회전
  transforms.CenterCrop(200), # 회전되어 추가된 영역이 들어가지 않도록 적당히 크기 조절할 것
  transforms.Grayscale(), # RGB로 표현되는 색깔이 흑백으로만 처리되어 그림의 데이터 차원이 줄어듦
  transforms.RandomVerticalFlips(), # 그림을 세로로 랜덤으로 뒤집기
  transforms.ToTensor(), # 모델에 들어갈 데이터 형식으로 맞추기 
  transforms.Normalize(means=[0.5], std=[0.5]) # 해당 과정은 모델의 훈련 효과를 높이기 위해 적용되는 것
])
x = preprocess_steps(img)
x.size

x.mean(), x.std()
