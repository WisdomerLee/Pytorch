# Image Preprocessing
Orininal image - pixel들로 구성
Dimensions(C, H, W)color, height, width
color에서 3가지는 RGB 값, 
흑백일 경우 1가지
height는 그림의 세로 픽셀 갯수
width는 그림의 가로 픽셀 갯수

pixel은 0~255까지의 값을 갖고 있음

PyTorch는 Image conversion에 사용

# Resize
그림의 크기(scale)를 조절해야 함 - 일반적으로 크기를 줄여서 활용
모델에 들어가는 입력은 모두 같은 크기의 tensor 모습을 가져야 함 (tensor shape가 동일해야 함)


# CenterCrop
그림을 가운데를 기준으로 특정한 크기로 자르기

# Grayscale
컬러 사진을 흑백 사진으로 변경
color channel의 차원이 3에서 1로 줄어듦

# RandomRotation
그림을 가장자리와 함께 회전, 일반적으로 crop과 함께 결합해서 사용함
그림에 없던 부분 등을 잘라내야 하기 때문

# RandomVerticalFlip
그림을 확률적으로 위아래를 뒤집음

# ToTensor
PIL image -> (C, H, W)의 차원을 갖는 tensor, 또한 tensor에 들어가는 값들은 0과 1사이

# Normalize
tensor로 변환된 그림을 평균, 표준편차가 같도록 데이터를 변환하는 것
Batch of Images (그림의 일정 묶음 단위에서)
평균과 표준편차 값이 같아지도록 그림의 데이터를 조정하는 것

# Compose
여러 transformation을 한 번에 실행하는 것
모든 그림에 composed step이 적용됨
```
preprocess_steps = transforms.Compose([
  transforms.Resize(300), # (300, 300)으로 지정해도 됨
  transforms.RandomRotation(50),
  transforms.CenterCrop(500),
  transforms.GrayScale(),
  transforms.RandomVerticalFlip(),
  transforms.ToTensor(),
  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 여기에 쓰이는 숫자는 연구를 통해 학습에 가장 좋게 영향을 준 값을 찾아낸 것으로, normalize에서 해당 값은 그대로 사용될 예정
])
x = preprocess_steps(img)
```

