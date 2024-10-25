
from detecto import core, utils
from detecto.visualize import show_labeled_image
from torchvision import transforms
import numpy as np


path_images = 'images'
path_train_labels = 'train_labels'
path_test_labels = 'test_labels'
# 사물 감지에서 label은 단순히 텍스트만 필요한 것이 아니라, 어느 사물이 있고, 어느 위치에 있는지에 대한 정보가 모두 표시되어야 함, 문서는 xml 파일로 존재

custom_transforms = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((50)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(15), # 최대 15도까지 각도 틀기
  transform.ToTensor(),
  utils.normalize_transform()
])

trained_labels = ['apple', 'banana']

train_dataset = core.Dataset(image_folder=path_images, label_data=path_train_lables, transform=custom_transforms)
test_dataset = core.Dataset(image_folder=path_images, label_datapath_test_lables, transform=custom_transforms)

train_loader = core.DataLoader(train_dataset, batch_size=2, shuffle=False)
test_loader = core.DataLoader(test_dataset, batch_size=2, shuffle=False)

model = core.Model(trained_lables)

model.get_internal_model()

losses = model.fit(train_loader, test_dataset, epochs=2, verbose=True)

test_image_path = '.jpg' # 훈련이 잘 되었는지 임의의 파일 하나를 골라서..
test_image = utils.read_image(test_image_path)
pred = model.predict(test_image)
labels, boxes, scores = pred
show_labeled_image(test_image, boxes, labels)  # 이것을 확인해보면 예측한 것들이 매우 많이 엉켜있어 사람이 알아보기 어려움

# 특정 이상의 IoU를 갖는 것들을 혹은 정확성이 70%이상이라고 추정되는 것들만 표시되게 해본다면?
conf_threshold = 0.7
filtered_indices = np.where(scores > conf_threshold)
filteres_scores = scores[filtered_indices]
filtered_boxes = boxes[filtered_indices]
num_list = filtered_indices[0].tolist()
filtered_labels = [labels[i] for i in num_list]
show_labeled_image(test_image, filtered_boxes, filtered_lables)

