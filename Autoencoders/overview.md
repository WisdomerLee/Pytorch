# Autoencoders가 무엇?

Neural Network는 입력 node, 숨은 node, 출력 node가 있음
그 중에
Encoder라고 부르는 부분은 입력 node에서 숨은 node로 넘어가는 과정까지
Decoder라고 부르는 부분은 숨은 node에서 출력 node로 넘어가는 과정까지를 가리킴

Autoencoder는
입력 데이터의 표현 방식을 배움!
비지도 학습에 해당
사용되는 곳은 

- 차원을 줄이거나
- 노이즈를 줄이거나

input neuron과 output neuron은 동일 (갯수, 차원 등)
모델 훈련: hidden layer는 input layer의 표현을 배움

모델 어플리케이션: encoder는 data의 차원을 보다 줄이고자 할 때 쓰임

# Shallow and Deep Autoencoders
숨은 layer가 한 개일 수도, 숨은 layer가 여럿일 수도 있음

# 적용
얼굴 인식에 활용 가능
차원 감소, 압축에 활용 가능
일반적이지 않은 것 감지
그림의 잡티 제거

# Coding
기본 그림이 있고(64, 64, 1)의 64, 64크기의 흑백 사진
그것을 128의 1차원으로 (Latent Space)
다시 되돌리기 - latent space의 내용을 다시 이전의 이미지로 되돌리기 기존의 64, 64, 1의 차원으로

