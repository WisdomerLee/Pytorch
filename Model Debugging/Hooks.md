# CNN을 hooks로 디버그하기
Hooks라는 것은 강력한 디버그, 상황을 파악하는데 좋은 도구
neural network 내부 에서 어떤 일이 일어나는지 이해할 수 있음
forward/backward pass가 진행될 때 같이 코드를 동작시킬 수 있음!
tensor, module, network등 어디에도 함수를 붙일 수 있음
neural network가 실행되면, hook도 실행

Forward Hook - forward pass 진행될 때 같이 실행
Backward Hook - backward pass 진행될 때 같이 실행

# Hook 집어넣기
1. Hook class 정의
2. Hook 등록
3. Neural Network 실행
4. Hook 실행 결과 확인

1에 해당되는 부분은 아래와 같이 초기화와 call함수에 layer를 쌓는 부분을 정의할 것
```
class HookExample:
  def __init__(self):
    self.layer_out=[]
    self.layer_shape=[]

  def __call__(self, module, module_in, module_out):
    self.layer_out.append(module_out)
    self.layer_shape.append(module_out.shape)
```

2에 해당되는 부분은 아래와 같이, 아래의 예시는 Conv2d를 통과하는 부분에 forward로 진행될 때 실행될 hook를 등록
```
hook = HookExample()
for l in model.modules():
  if isinstance(l, torch.nn.modules.conv.Conv2d):
    handle = l.register_forward_hook(hook)
```
3에 해당되는 부분은 코드에서 
```
y_pred = model(X)
```
4에 해당되는 부분 - hook에 있는 내용 확인할 것
```
layer_num = 0
layer_imgs = hook.layer_out[layer_num].detach().numpy()
```

# Hook Coding
CNN의 실행에서는 Layer out은 그림의 결과가 나오고
Layer shape 은 layer를 통과하면서 입력된 데이터의 크기(차원)가 어떻게 변화하는지 볼 수 있음

