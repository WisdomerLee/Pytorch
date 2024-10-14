Neural Network

# forward pass

훈련 데이터 -> 입력 layer -> Activation layer -> prediction data
실제 데이터/ 예측 데이터 -> loss function -> loss score -> optimizer -> layer weights 업데이트

weight를 어떻게 업데이트를 해야 하는가??

# Weight Update
loss score를 계산한 값을 확인하기
loss 값을 계산하는 함수의 gradation을 확인
오차의 값의 gradation이 0이 되는 지점으로 이동되도록 (gradaient descent)
다만 gradient만 추가하는 것이 아니라, 일부 보정 값을 추가하는데, 이 값이 learning rate

미분 계산할 때 chain rule을 이용할 것
y가 x에 의존적이고, z는 y에 의존적이라면
x의 값의 변화에 z의 변동량(미분 값)은 x의 값 변화에 y의 값의 변화 * y값의 변화에 z 값의 변화와 동일한 chain rule을 이용하여 미분을 계산할 것

# dot product
벡터의 값을 내적하는 것
어느 벡터에 더 가까운가를 판별하는데 좋은 기준점이 됨
값이 비슷하면 크기, 각도가 비슷해짐

실제 neural network로 입력되는 값들은 모두 vector로 환산되어 들어가므로, 
wegith는 입력데이터, 출력 데이터에 적용되어야 하고
x와 유사한 값을 dot product로 찾게 됨
