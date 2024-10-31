# 알아야 할 것
Pretrained Model
Transfer Learning

DensNet121이라는 이미 훈련된 모델을 활용하여, 개, 고양이 구분으로 활용할 예정

# 왜 이런 것이 도입되었는가??
훈련시키기 위해 활용될 그림 데이터 같은 것이 적음
이미 훈련된 모델을 활용한다면...!
이미 기존에 활용되던 weights를 쓸 수 있고
다른 클래스에 있는 이미 학습된 features들을 이용할 수 있으며
같은 성능에 도달하기 위한 시간을 줄이고, 더 나은 성능을 쉽게 찾을 수 있음
또한 이미 테스트된 구조이므로 안정적임

# 활용방식
Input layer -> Convolutional Base -> Source Labels

transfer learning은
Convolutional Base는 그대로 두고, 입력 layer, new labels을 바꾸기
즉 입력, 출력만 바꿔치기하는 것

# Imagenet
Imagenet은 다양한 사물을 구분하기 위한 도전으로 만들어진 neural network
2010년부터 개발되었으며, 매년 업데이트가 진행됨
1000개의 카테고리가 있고 훈련에는 1백만이 넘는 그림이 활용되었음

# VGG19
Deep Convolutional Network 
Imagenet 도전을 위해 개발된 모델 중 하나

