# GAN은??
deep learning network의 종류 중에 하나
두 가지 neural network가 있음

- Generator : 가짜 이미지를 생성
- Discriminator : 가짜인지, 진짜인지 판별

Discriminator는 가짜인지, 진짜인지 구분하려 하고
Generator는 진짜같은 가짜 이미지를 만들려고 함

각 network는 하나의 시스템으로 연결되어있으며, zero-sum 게임으로 경쟁을 하게 됨
두 network가 동시에 훈련

# 훈련은 어떻게?
초반 훈련:
Discriminator가 진짜, 가짜를 쉽게 구분함
Generator의 성능이 좋지 않음
후반 훈련:
Discriminator가 진짜, 가짜를 구분하기 어려워짐
Generator의 성능이 좋아짐

# Architecture

Random Input -> Generator로 들어가서 sample이 생성,
실제 그림이 sample로 생성
Discriminator는 sample이 참, 거짓인지 구분
Discriminator loss, Generator loss

# Discriminator
Discriminator는 sample로 들어오는 데이터가 참인지 거짓인지 판단
훈련 데이터로는 실제 사진, Generator로 만들어진 사진이 쓰임
Discriminator loss는
Discriminator가 잘못 판별한 것에 비례하여 해당 neural network에 페널티가 주어짐

# Generator
보다 정밀하게 위조된 그림을 그리기 위해 즉, Discriminator가 속을 정도로 정교하게 훈련을 진행함
Discriminator를 고려
Generator loss는 Discriminator에 의존

# Generator의 훈련
1. 무작위 데이터 생성
2. Generator의 forward output 생성
3. Discriminator의 결과 얻음
4. Generator loss 계산
5. Generator loss를 기반으로 backpropagation 진행
6. Generator weight 업데이트

# 훈련
두 network가 훈련
각 network가 배우는 속도는 비슷해야 함, 한쪽이 너무 빨리 배우면 최종 결과가 좋지 않음

훈련할 때 epoch마다 아래의 과정이 반복
Discriminator를 몇 epoch에서 훈련하고, Generator를 몇 epoch에서 훈련하는 형태가 진행

Generator는 Discriminator가 훈련할 때 Generator는 weight의 update가 없음
Discriminator는 Generator가 훈련할 때 Discriminator는 weight의 update가 없음

목적 - Generator의 성능이 좋아지고, Discriminator의 성능이 낮아지게 하는 것
이론적으로 Discriminator의 정확도는 50%로 가야 함(완벽하게 추측하게 하는 것)

# 활용
얼굴 생성 모델
해당 알고리즘이 보다 정교화 되었고, 컴퓨터 성능이 좋아진 것이 큰 영향을 줌

약 - 희귀 질병에 대한 데이터
비디오 - Deep fake, 비디오의 프레임 속도 높이기 - Dual Video Discriminator GAN
음악
그림 - SkeGAN, GANpaint
말 - GAN based text-to-speech(GAN-TTS)
로봇

# 장단점
장점
흥미로움
Image upsampling(실제 그림보다 더 고해상도가 될 수 있음)
그림, 음악, 비디오, 미술 생성에 활용
단점
훈련이 쉽지 않음
컴퓨터 연산 자원 소모가 많음
Deepfakes (사회적인 부작용)

