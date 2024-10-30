# Style Transfer

Style Transfer model을 어떻게 만들 수 있는가?
그리고 그 model을 만드는 코드


# Intro
Gatys, Ecker, Bethge "A Neural Algorithm of Artistic Style"
Source: https://arxiv.org/pdf/1508.06576.pdf

Content Image 
Style Image

두 가지가
Style Transfer Model로 전달되어
Content Image에 Style Image의 양식이 전달!

# 이미 훈련된 모델
VGG16, VGG19등을 사용할 수 있음
CNN은 훈련동안 바뀌지 않음 << 다른 훈련과 몹시 다른 차이라는 것을 반드시 기억할 것
artistic image만 변경


일반적인 Deep Learning 훈련은
Input -> Model -> Target
        weights updates

Style Transfer Training
Input -> Model -> Target
        weights frozen

Target이 변경...

# 모델 특징
VGG19
16개의 convolutional layers
5개의 pooling layers

그림을 이루는 요소는 결국 작은 픽셀, 선, 가장자리 에서 출발하여 요소, 사물과 같은 것이 됨

기존의 CNN의 첫layer들을 style feature extraction으로 활용
그리고, content의 feature extraction으로는 마지막 Convolutional쪽의 두 번째 layer를 사용

# Losses
손실 계산방식
Content Image -> 생성된 Image <- Style Image

이미 훈련된 Network(CNN의 weights update는 없음!) - CNN 쪽이 아닌 쪽의 layer들이 update
content image와 생성된 image에서
content features를 추출 - content loss
style image와 생성된 image에서
style features를 추출 - style loss

total loss = content loss, style loss모두

# Feature Maps
Feature maps는 무엇??
이미지의 영역을 잘게 쪼개고, 그 영역에서 추출한 값들의 집합!!! (즉 feature map이 여럿 쌓여있음) - 대각선 왼쪽 윗 상단의 feature, 그 옆의 feature, ...
그렇게 추출하는 것들은 선이 될 수도, 특정 도형을 닮은 모습일 수도 있음!

# Style?
사용되는 색깔들
색깔 분포
공통으로 사용되는 표현 같은 것들

# Feature maps와 연관성은?
알고리즘이 추출하는 것
Feature map 간의 관계성
Feature maps Matrix가 있고
Transposed Feature Maps Matrix (Matrix의 Transpose- 대각성분 기준으로 대칭된 변환)가 있는데
그 둘을 곱한 결과가 Gram Matrix가 됨

loss 값을 계산할 때 활용
