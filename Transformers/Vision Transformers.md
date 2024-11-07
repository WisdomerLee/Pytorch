# Vision Transformers (ViT)

Pixel이 기본 단위
pixel간의 관계는 특징지을 수 없음
image sections analyzed (positional encoding)
그림을 잘게 쪼개어 분석 (나눈 부분의 위치도 같이 encoding) 각 쪼개진 부분별로 분석
쪼개진 section간 관계를 계산
Patch size - 16x16 : section이 어느 크기 단위로 쪼개지는가?
Stride - 16x16 : 한 section에서 다음 section으로 어느 크기로 넘어가는가? - 대체로 Patch size와 같게 설정하는 편, 보다 작게 설정하면 section에서 overlap이 발생
overlap 허용

그림을 쪼갤 때는 모두 같은 크기여야 함!
그리고 쪼개진 영역(section)은 flatten 과정을 거침
또한 각 쪼개진 영역은 몇 번째에 해당되는지 포함된 positional encoding이 추가로 들어감
이 변경 과정에서 1차원의 tensor로 변경
이 1차원의 tensor들이 
CLS token이 되고
multi-head self-attention, dense의 encoder layer를 지나 hidden layer - softmax activation -> output layer -> 그림 분류 예측

ViT는 CNN보다 성능이 좋으나, 100M보다 많은 이미지들에서 성능이 더 좋은 편
JFT(Imagenet dataset with up to 300 M images, 18000 classes) - 그런데... 비공개 dataset

대형 모델에서도 성능이 계속 잘 올라가는 편
