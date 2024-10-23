from typing import OrderedDict
import torch
import torch.nn as nn


input = torch.rand((1, 3, 32, 32))

# 아래와 같은 방식으로도 모델을 생성할 수 있음

model = nn.Sequential(OrderedDict([
  ('conv1', nn.Conv2d(3, 8, 3)), # 해당 출력은 bs, 8, 30, 30 - 입력 채널 3개, 출력 채널 8개, kernel_size 3 - 일반적으로 커널 크기는 3 혹은 5를 주로 씀, 컬러 사진의 경우 
  ('relu1', nn.ReLU()),
  ('pool', nn.MaxPool2d(2,2)), # 출력은 bs, 8, 15, 15
  ('conv2', nn.Conv2d(8, 16, 3)), # 출력은 bs, 16, 13, 13
  ('relu2', nn.ReLU()),
  ('pool2', nn.MaxPool2d(2,2)), # 출력은 bs, 16, 6, 6
  ('flatten', nn.Flatten()), # 출력은 (3, 16 * 6 * 6) 
  ('fc1', nn.Linear(16*6*6, 128)),
  ('relu3', nn.ReLU()),
  ('fc2', nn.Linear(128, 64)),
  ('relu4', nn.ReLU()),
  ('fc3', nn.Linear(64, 1)),
  ('sigmoid', nn.Sigmoid())
]))

# 각 레이어 별로 주석처리하여 한 단계 거치면서 아래의 코드로 출력의 shape를 살펴볼 것
model(input).shape
