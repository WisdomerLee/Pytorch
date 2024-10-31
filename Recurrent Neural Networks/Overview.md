# RNN
RNN은 데이터가 순차적(Sequential) 일 때 좋은 선택

시간에 따라 들어오는 데이터를 처리하거나
Natural Language Processing
Speech Recognition

# Rolled and Unrolled RNN
RNN을 바라보는 두 가지 관점 : rolled, unrolled

rolled는 neuron에서 낸 결과가 다시 같은 neuron의 입력으로도 들어가는 것
unrolled는 neuron의 결과가 다른 neuron에 영향을 주는 것

# Sequences Input, Hidden State, Output
입력도 하나, 출력도 하나일 수 있음
하나의 입력이 여러 neuron으로 나뉘어 출력이 여럿이 될 수 있음
여러 입력이 하나의 출력으로 나올 수도 있음
여러 입력이 여러 출력으로 나올 수도 있음

# 기본 RNN과 LSTM의 차이
기본 RNN cell
hidden neuron이 다른 neuron과 연결되어(혹은 input의 여러 노드 중에서 직접적으로 올 수도 있음) (tanh와 같은 함수등을 이용하여 해당 neuron의 값을 보정하여 추가로 넣음) 출력 계산에 영향

LSTM cell
다른 neuron에서 받아들이는 값을 보정하는 방식이 조금 더 복잡해지는데, 다른 곳에서 주는 영향을 일정 기간 '기억'하는 보정이 추가로 들어감

# LSTM cell
https://colah.github.io/posts/2015-08-Understanding-LSTMs/

내부에 게이트가 있음
Forget Gate : cell(neuron)의 정보를 기억할지, 잊어버릴지를 결정
Input Gate Layer & State Update - 새로 들어온 정보를 이용하여 셀에 저장할 데이터를 선택
output - 다음 상태, 다음 레이어로 상태 전달

# Practical considerations: Input shape
데이터가 3차원의 shape을 가져야 할 때 - 일반적으로 RNN이 다루는 정보는 3차원이 기본
sample, 시간 스텝 간격, features

multi-variate prediction - 다중 변수 데이터가 있게 된다면???
batch_size(sample), seq_len(시간 스텝), features(input_size)

# RNN의 장점, 단점
sequential data를 다룰 때 매우 좋음
LSTM은 memory를 조금 더 길게 가질 수 있음
기본 RNN은 매우 간단하고, 숫자 계산에서 연산이 오래 걸리게 됨??
