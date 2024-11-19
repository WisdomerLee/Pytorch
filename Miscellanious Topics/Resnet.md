# Resnet

# 문제
그림 과 같은 것들을 보다 복잡한 작업을 수행하도록 하고 싶음 함
해결 방식으로 layers를 더 추가하기!

# Deeper Networks의 효과
예상으로는 layers와 parameters들을 많이 추가하면 성능이 좋아질 것으로 예상하였음
현실은
deeper network는 훈련이 더 힘들고 (gradient가 layer로 전파되는데, 더 깊은 곳으로 갈 수록 gradient의 영향이 사라져가거나, 혹은 폭발적으로 발산할 가능성이 증가), 오히려 성능이 감소하는 구간이 발생함

그럼 어떻게 해결해야 하는가?? - Skip connections
connection 중에 일부를 넘기고 듬성듬성 연결하는 방식을 채택 함 - neuron의 연결과 흡사


# Skip Connections
Shortcut Connections, residual connection 모두 같은 말
성능 개선과 일반화에 큰 도움이 됨
gradient 전파 시 하나 혹은 더 많은 레이어를 건너뛰고 전파 할 수 있음
deep architectures에서 종종 활용됨

목적
gradient의 영향이 점차 감소하는 문제를 극복
과적합을 예방
network의 학습을 보다 쉽게 함
성능의 개선

image classification, language translation, speech recognition 등에 활용될 수 있음

# 성능(Performance)에 미치는 영향
같은 층을 놓고 비교한 모델에서 더 나은 성능을 보임

# Resnet과 다른 모델과 비교
ResNet-152
VGG-16, VGG-19
ENet, GoogLeNet, 

상대적으로 파라미터의 크기에 비해 높은 성능을 보임

