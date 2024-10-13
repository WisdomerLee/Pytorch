훈련 데이터와 평가 데이터, 테스트 데이터 분할하는 방식

모델 훈련에 사용되는 데이터, 유효성을 검증하는데 활용되는 데이터, 모델의 예측값을 실제로 확인하는 데이터 등으로 나눔

## Training data - 모델 훈련에 사용됨
## Validation data - 모델 평가에 사용됨, fine-tune model의 hyperparameter들을 사용,
모델은 종종 이 데이터를 보지만, 배우는데 활용되진 않음, 모델에 간접적으로 영향을 줌

## Test data - 최종 모델의 평가에 사용
표준 제공
모델은 이 데이터를 오직 보기만 함
경쟁 모델의 평가에 활용되기도 함
validation data와 같은 분산을 갖고 있음
종종 training data와 validation data만 활용되고, test data는 없는 형태로 진행되기도 함

# 분할 비율
두가지 요소 - 총 샘플의 갯수, 실제 모델
몇몇 모델은 훈련 데이터가 더 필요한 경우가 있음
validation data는 모델간 차이점을 확인할 만큼 크면 됨
hyperparameter가 적은 모델은 쉽게 평가 가능 - validation dataset의 크기가 작아도 됨
hyperparameter가 큰 몬델은 평가가 쉽지 않음 - validation dataset의 크기가 커야 함

