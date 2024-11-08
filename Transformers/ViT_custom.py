
# source: https://medium.com@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-eodoscopists-in-under-5-min-2e7e4110a353
# hugsvision을 사용하기 위해서는 package 설치가 필요함
# https://github.com/qanastek/HugsVision에서 설치 방법 확인 필요


from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInerference

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# data 준비
train, val, id2label, label2id = VisionDataset.fromImageFolder(
  "./train/",
  test_ratio = 0.1, # 테스트 데이터 비율
  balanced = True, # 
  augmentation = True, # 데이터 증강
  torch_vision = False
)
# 모델 가져오기
huggingface_model = 'google/vit-base-patch16-224-in21k'

# model 가져오기! > 이미 훈련된 모델 가져오기 + 모델 훈련까지 같이 하는 듯...?
trainer = VisionClassifierTrainer(
  model_name = "MyDogClassifier",
  train = train,
  test = val,
  output_dir = "./out/",
  max_epochs = 20,
  batch_size = 4,
  lr= 2e-5,
  fp16 = True,
  model = ViTForImageClassification.from_pretrained(
    huggingface_model,
    num_labels = len(label2id),
    label2id = label2id,
    id2label = id2label
  ),
  feature_extractor = ViTFeatureExtractor.from_pretrained(
    hugging_face_model
  ),
)

# 모델 평가

y_true, y_pred = trainer.evaluate_f1_score()

cm = confusion_matrix(y_true, y_pred)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index= labels, columns = labels)

# plt.figure(figsize = (10, 7))

sns.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig("./conf_matrix_1.jpg")

import os.path
path = "./out/MYDOGCLASSIFIER/20_2022-09-02-13-14-14/model/" # 참고로 model앞에 붙은 저것은 실행 시간과 일치한다고 함 > folder를 보고 매번 model의 폴더 위치 확인할 것
# 또한 모델을 제대로 불러오려면 model 폴더에서 config.json을 복사해두고 복사한 파일을 preprocessor_config.json으로 변경해야 함 > 미리 훈련된 모델일 경우
img = "./test/affenpinscher/affenpinscher_0.jpg"

classifier = VisionClassiferInference(
  feature_extractor = ViTFeatureExtractor.from_pretrained(path),
  model = ViTForImageClassification.from_pretrained(path)
)

label = classifier.predict(img_path=img)
print("Predicted class: ", label)

# Test dataset
test, _, id2label, label2id = VisionDataset.fromImageFolder(
  "./test/",
  test_ratio = 0,
  balanced = True,
  augmentation = True,
  torch_vision = False
)

classifier.predict(img)

import glob
test_files = [f for f in glob.glob("./test/**/**", recursive=True) if os.path.isfile(f)]

for i in range(9):
  print(f"{test_files[i]}")
  print(f"predicted: {classifier.predict(test_files[i])}")

