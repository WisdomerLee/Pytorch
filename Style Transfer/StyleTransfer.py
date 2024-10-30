
import numpy as np
from PIL import Image
import torch
from torch.optim import Adam
from torchvision import transforms
from torch.nn.functional import mse_loss
from torchvision import models

vgg = models.vgg19(pretrained=True).features
print(vgg) # vgg의 layer들을 확인할 것 - 아래의 LOSS_LAYERS에서 layer들을 활용하는데 사용될 layer층을 지정하는데 해당 정보 필요
# 기존의 이미지 분류에서와 달리 흑백으로 변환하거나 하는 부분은 없음! color의 분포 역시 style의 중요한 특성이기 때문
preprocess_steps = transforms.Compose([
  transforms.Resize((200, 200)),
  transforms.ToTensor(),
])

content_img = Image.open('Hamburg.jpg').convert('RGB') # 혹시라도 있을 alpha 채널을 없애고, RGB 채널을 가진 그림으로 변환
content_img = preprocess_steps(content_img)
# 아래의 것은 C, H, W로 된 값을 H, W, C의 순서로 바꾸는 것 - 입출력을 맞추기 위함
# content_img = content_img.transpose(0, 2)
content_img = torch.unsqueeze(content_img, 0)
print(content_img.shape)

style_img = Image.open('The_~.jpg').convert('RGB')
style_img = preprocess_steps(style_img)
# style_img = style_img.transpose(0, 2)
style_img = torch.unsqueeze(style_img, 0)
print(style_img.shape)

LOSS_LAYERS = { '0': 'conv1_1',
               '5': 'conv2_1',
               '10': 'conv3_1',
               '19': 'conv4_1',
               '21': 'conv4_2',
               '28': 'conv5_1'
}

def extract_features(x, model):
  features = {}
  # 특징 추출을 위해 모델의 layer를 따라가다가
  for name, layer in model_modules.items():
    x = layer(x)
    # LOSS_LAYERS로 정의한 layer에 도달하면, 해당 layer 층의 값을 loss 계산을 위한 feature로 뽑아내기
    if name in LOSS_LAYERS:
      features[LOSS_LAYERS[name]] = x
  
  return x

content_img_features = extract_features(content_img, vgg)
style_img_features = extract_features(style_img, vgg)

def calc_gram_matrix(tensor):
  _, C, H, W = tensor.size() # tensor에서 색깔, 높이, 너비를 알려주는 부분을 찾아내고
  tensor = tensor.view(C, H * W) # tensor의 속성에서 색깔, 높이*너비의 tensor로 변환
  gram_matrix = torch.mm(tensor, tensor.t()) # 그리고 featuremaps로 나와있는 matrix와 해당 transpose()matrix를 곱
  gram_matrix = gram_matrix.div(C*H*W) # normalization
  return gram_matrix

style_features_gram_matrix = {layer: calc_gram_matrix(style_img_features[layer]) for layer in style_img_features} # 위에서 뽑았던 features들의 layer를 이용하여 gram_matrix 계산
print(style_features_gram_matrix)

weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.6, 'conv4_1': 0.4, 'conv5_1': 0.2}

target = content_img.clone().requires_grad_(True)

optimizer = Adam([target], lr=0.003)

for i in range(1, 100):
  
  target_features = extract_features(target, vgg)

  content_loss = mse_loss(target_features['conv4_2'], content_img_features['conv4_2'])

  style_loss = 0
  for layer in weights:
    target_feature = target_features[layer]
    target_gram_matrix = calc_gram_matrix(target_feature)
    style_gram_matrix = style_features_gram_matrix[layer]

    layer_loss = mse_loss(target_gram_matrix, style_gram_matrix) * weights[layer]
    # layer_loss *= weights[layer]
    style_loss += layer_loss
  total_loss = 1000000 * style_loss + content_loss # 일반적으로 content loss가 style loss보다 백만배 가까이 큰 편이라고 함, 차이가 너무 크면 style loss는 영향력이 사실상 없어지는 형태라 이와 같이 보정을 진행
  if i % 10 == 0:
    print(f"Epoch: {i}, Style Loss: {style_loss}, Content Loss {content_loss}")

  optimizer.zero_grad()
  
  total_loss.backward(retain_graph=True)

  optimizer.step()

mean = (0.485, 0.456, 0.406) # imagenet mean and std - 이 것은 실험을 통해 가장 좋은 값으로 찾아낸 값이라고 함
std = (0.229, 0.224, 0.225)

def tensor_to_image(tensor):
  image = tensor.clone().detach()
  image = image.cpu().numpy().squeeze()
  image = image.transpose(1, 2, 0) # 그림 파일의 차원에 맞게 tensor 재조정
  image *= np.array(std) + np.array(mean)
  image = image.clip(0, 1)
  return image

import matplotlib.pyplot as plt
img = tensor_to_image(content_img)

fig = plt.figure()
fig.suptitle('Target Image')
plt.imshow(img)
