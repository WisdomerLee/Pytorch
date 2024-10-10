
모델 생성 진행

1. 모델 훈련 데이터 준비
2. Neural Network Layers로 입력 데이터 전달
3. 각 Layer의 비중인 Weights가 보정용으로 사용 - 훈련 시에 해당 weights들이 변경
4. 모델이 예측한 데이터가 나옴
5. 실제 데이터와 비교
6. 두 데이터를 비교하여 Loss Function으로 오차 계산 - Loss Score 값 나옴
7. Optimizer가 Loss Score를 받아들여 Weights에 업데이트

예측은 어떻게 진행되는가?
입력 데이터가 있으면
Neural Network로 전달되어
각 Layer의 Weights들과 layer의 파라미터 값들이 소폭 보정된 값을 토대로
값을 예측
실제 데이터와 비교하여 모델을 평가

