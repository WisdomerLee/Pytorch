# parameter를 덧대서 고치는 이유
훈련, 추론 시간을 늘리거나, 줄이기 위함
결과를 개선
훈련 수렴

# 조정 가능한 Hyperparameter
Hyper-parameter는 모델의 성능에 큰 영향을 줌
parameter의 여러 조합을 확인하면 최고의 성능을 얻을 수 있음
이를 가능하게 하는 packages는 RayTune, Optuna, skorch 등이 있음
hyperparameter는 network topology(신경망의 짜임새), node 숫자, layer type, activation function등등을 지정하는 것등 매우 다양함

# Batch size
모델을 훈련시킬 때 데이터를 나눌 단위의 크기를 지정해주는 하이퍼 파라미터

작은 batch size는 GPU의 유용성이 줄고, 반복되는 횟수가 늘어나지만, 훈련 안정성이 증가
큰 batch size는 GPU의 유용성이 늘고, 반복되는 횟수가 줄어드나, 훈련 안정성이 감소
2의 배수로 숫자를 지정할 것
기본적으로 32의 값을 기본으로 사용

# Epochs
훈련을 몇 번 반복시킬 것인가!를 지정하는 파라미터
모든 데이터를 한 번 훈련시키는 단위- epochs
이 숫자가 낮으면
훈련 시간이 짧아지고, 모델 성능은 낮아짐, 다만 안정성은 높아짐
이 숫자가 높으면
훈련 시간이 길어지고, 모델 성능은 증가(다만 과하면 점차 낮아짐), 안정성 낮아질 수 있음

# Hidden Layers
숨어있는 학습에 관여하는 겹, 층의 갯수를 결정하는 파라미터
입력과 출력 사이에 있는데, 이 겹이 늘어날 수록 모델이 학습하는 것이 조금 더 복잡하고 정교한 패턴을 익힐 수 있음
훈련 시간은 층의 갯수가 적으면 적고, 많으면 오래 걸림
평가 시간 역시 마찬가지
과적합의 위험성은 층의 갯수와 반비례

# Nodes within a Layer
겹 자체가 얼마나 많은 perceptron(node)를 갖고 있을지를 결정하는 파라미터
역시 갯수가 늘어날 수록 모델이 복잡한 패턴을 학습하게 됨
훈련 시간, 평가 시간 모두 이 숫자에 비례
과적합의 위험성은 이 숫자와 반비례

# Types of Search
grid search
한정된 변수들로 파라미터들 설정하는 공간 만들 방식
가능한 조합의 각 평가
batch_size,와 learning rate의 가능한 조합등을 이용하여 모델을 평가
잘 알려진 파라미터들을 확인할 때 좋다고 함

random search
모델 설정할 때 하이퍼 파라미터들을 랜덤하게 설정하고 모델 훈련을 진행
새 발견을 위해서 좋은 방식

# skorch
scikit-learn 호환되는 neural network library
https://github.com/skorch-dev/skorch
scikitlearn을 pytorch 버전으로 변경

다음의 환경과 같이 사용할 수 있음
sklearn pipeline
grid search
