import torch
import torch.nn as nn

class MultiClassNet(nn.Module):
  # 애초에 분류할 모델이 3가지 밖에 되지 않는 형태이므로, 모델의 deep net도 굉장히 작게 구성됨
  # lin1, lin2, log_softmax

  def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
    super().__init__()
    self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES) # layer를 만들 때 전 layer의 출력부분에 해당되는 숫자가 다음 layer의 입력 부분에 해당되는 숫자와 반드시 일치해야 함을 기억하기
    self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES) # 또한 여기서 사용하는 layer는 선형의 관계를 갖는 형태로 간단한 layer임 - Linear - 마지막 layer의 출력은 반드시 구분되어야 할 종류의 숫자와 일치해야 함!!!
    self.log_softmax = nn.LogSoftmax(dim=1) # 마지막은 항상 activation function을 담당하는 layer를 둘 것 - logsoftmax라는 함수 사용

  def forward(self, x):
    x = self.lin1(x) # 첫 번째 layer 통과
    x = torch.sigmoid(x) # 첫번째 layer의 activation function 통과 - layer 하나를 통과할 때마다 activation function을 지정해야 함
    x = self.lin2(x) # 두 번째 layer 통과
    x = self.log_softmax(x) # 두 번째 layer의 activation function 통과 - layer가 2개이므로 마지막 activation function - 
    return x
