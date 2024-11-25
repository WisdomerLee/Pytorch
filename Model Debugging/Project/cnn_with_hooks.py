
import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 그림 불러오기
image_path = "kiki.jpg"
image = Image.open(image_path)

# 파일 전처리
transformations = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor()
])


X = transformations(image).unsqueeze(0)
X.shape

# 모델을 처음부터 훈련할 필요없이 이미 훈련된 모델을 같이 불러오기
model = resnet18(pretrained = True)

# Hook 설정하기
class CustomHook:

  def __init__(self):
    # layer output 저장할 것 모으기
    self.layer_out = []
    # layer shape 저장할 것 모으기
    self.layer_shape = []

  # 아래의 함수에서 module은 layer단위라고 생각하면 되고, module_in은 해당 layer에 들어가는 입력, module_out은 해당 layer에서 출력하는 것
  def __call__(self, module, module_in, module_out):
    self.layer_out.append(module_out)
    self.layer_shape.append(module_out.shape)

# Hook 등록하기
my_hook = CustomHook()

for l in model.modules():
  if isinstance(l, torch.nn.modules.conv.Conv2d): # 모든 layer에 전부 할당하지 않고 Conv2d에 해당되는 layer에만... > CNN의 가장 핵심 역할을 담당하는 layer이므로 그것에만 집중
    handle = l.register_forward_pass(my_hook)

# 모델로 예측을 실행해보기
y_pred = model(X)

# hook의 출력을 확인해보기
my_hook.layer_out
my_hook.layer_shape

# 보고 싶은 layer만 선택하고 싶다면...


layer_num = 0
layer_imgs = my_hook.layer_out[layer_num].detach().numpy()

for i in range(4):
  plt.imshow(layer_imgs[0, i, :, :])
  plt.show()

