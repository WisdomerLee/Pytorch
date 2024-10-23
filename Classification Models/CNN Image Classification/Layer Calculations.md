# Layer calculation

모델을 작성하고 코드를 실행할 때 가장 크게 많이 부딪히는 부분 중에 하나
layer와 layer 간에 입력, 출력이 서로 맞지 않아 (이전 layer의 출력이 다음 layer의 입력이 되어야 함)
에러를 발생시키는 경우가 매우 많음

# Tensor Dimensions
|차원|구조|사용|예시|
|------|---|---|---|
|1|batch_size|Lables/predictions|[16]|
|2|batch_size, features|nn.Linear()|[16, 512]|
|3|batch_size, channels, features|nn.Conv1d()|[16, 1, 512]|
|4|batch_size, channels, height, width|Conv2d()|[16, 1, 224, 224]|
|5|batch_size, channels, depth, height, width|nn.conv3d()|[16, 1, 5, 224, 224]|

# Conv2d Layers
해당 layer는 in_channels, out_channels가 필요
```
(class) Conv2d(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=1, padding: _size_2_t|str=0, dillation: _size_2_t = 1, groups: int=1, bias: bool= True, padding_mode: str = 'zeros', device: Any | None = None, dtype: Any | None = None)
```
|in_channels|out_channels|
|---|---|
|흑백: color channels - 1|convolution의 결과 생성될 channel의 숫자|
|칼라: color channels -3|kernel size에 의존, 해당 channel size는 다음 convolutional layer의 in_channel, 일반적으로 conv layers를 지날수록 channels의 숫자가 증가함|

# Transition from Convolutional Layer to Fully Connected
Conv2d는 4차원의 tensor를 가짐
fully connected - 2차원을 가짐
nn.Flatten()으로 될 수 있음
|Conv size|Fully Connected Size|
|---|---|
|[16, 48, 224, 224]|[16, 48*224*224]|

각각 앞 순서대로 batch_size, channel, height, width, batch_size, channel*height*width

# Fully Connected Layer
channel이 아닌, feature를 요구
in_features: input sample size
out_features: output sample size
```
(class) Linear(in_features: int, out_features: int, bias:bool=True, device: Any | None = None, dtype: Any | None = None)
```
- 이전 layer의 출력 feature가 다음 layer의 입력 feature가 됨을 기억
단, layer 사이에 있는 activation layer는 (입, 출력의 데이터 형태 변화에)아무런 영향이 없음을 기억할 것
  
