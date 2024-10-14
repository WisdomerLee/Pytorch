
PyTorch 구조는 variables로 구성되는데 그것이 PyTorch tensors
매우 간단한 수학적인 numpy 배열들인데, 매우 강력함
자동적으로 gradient를 계산
다른 tensor들에 얼마나 의존하는지를 (변화가 얼마나 크게 영향을 주는지를) 확인할 수 있음

# 자동으로 gradient를 계산
```
x = torch.tensor(1.0, requires_grad=True) # torch에서 tensor를 이용할 때 requires_grad=True라는 것을 지정하면,
# 첫번째 tensor에 의존하는 두 번째 tensor를 아래와 같이 정의하고
y = (x-3)* (x-6)* (x-4)

# 두 번째 tensor의 gradient 계산해보면
y.backward() 
print(x.grad) # 첫번째 tensor의 gradient 값이 자동으로 같이 계산
```

# Computational Graphs
간단한 network를 생각해보기
입력 x는 y를 계산하는데 활용되고, y는 z를 계산하는데 활용

이렇게 진행되는 것을 forward pass라고 부름

backpropagation
미분한 값을 토대로 역으로 z의 값의 변화를 토대로 y의 값이 얼마나 바뀔지, y의 값의 변화를 토대로 x의 값이 얼마나 바뀔지를 역으로 계산

## update of weights
입력을 토대로 최종 출력 값을 계산
실제 나와야 할 값이 있음
실제 나와야 할 값과 예측된 출력 값의 오차를 계산
weights는 node로 인식
optimizer는 backpropagation으로 계산된 미분 값을 토대로 weights를 업데이트!!!

# Computational Graphs - Forward Pass
complex network에서 여러 input을 받아 처리할 경우
의존하는 변수들 여럿을 서로 의존하지 않는 변수들끼리 묶어 하나의 Tensor로 구성!(입력 tensor)
그 tensor들에 의존하는 내부의 파라미터들을 역시 또다른 tensor의 파라미터들로 묶음!
반복
