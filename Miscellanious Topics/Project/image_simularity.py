# 그림의 유사성 테스트!
from datasets import load_dataset, list_datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import os

# 모델 불러오기
mode = models.resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1]) # 모델의 마지막 출력 부분을 제외한 모든 부분을 활용

model.eval()

# 그림 전처리
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 이 마지막 함수는 꼭 필요한 것은 아니나, 품질을 조금 더 높여준다고 함
])

# 그림 불러오기
image_path = '..그림이 들어있는 폴더 경로'
image_files = os.listdir(image_path)

img = Image.open(image_path+ image_files[0]).convert('RGB') # alpha 채널 없애기

preprocess(img).shape # 이상태는 batchsize가 포함되어있는데
preprocess(img).unsuqeeze(0).shape # 앞에 하나를 더 추가하여 해당 객체가 한 개임을 알려주는 차원 하나를 더 추가

# 연관 그림 embeddings 만들기
embeddings = []
for i in range(100):
  img = Image.open(iamge_path + image_files[i]).convert('RGB')
  img_tensor = preprocess(img).unsqueeze(0)

  with torch.no_grad():
    embedding = model(img_tensor)
    embedding = embedding[0, :, 0, 0] # 두번째 해당되는 부분만 추출!
  
  embeddings.append(embedding)
    

# target image와 embeddings 비교
sample_img = Image.open(image_path + image_files[101]).convert('RGB')

img = preprocess(sample_img).unsqueeze(0)
with torch.no_grad():
  sample_embedding = model(img)
  sample_embedding = sample_embedding[0, :, 0, 0]
  
# cosine similarity 계산
similarities = []
for i in range(len(embeddings)):
  # cosine similarity 계산
  similarity = torch.cosine_similarity(sample_embedding, embeddings[i], dim=0).tolist()
  # euclidean distance
  # similarity = torch.dist(sample_embedding, embeddings[i], p=2)
  similarities.append(similarity)

# 가장 비슷한 것을 찾아서 
idx_max_similarity = similarities.index(max(similarities))
# 어느 파일이 가장 가까운지 알려주기!
img_files[idx_max_similarity]
