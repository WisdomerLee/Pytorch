
import torch
import torch.nn as nn

class ImageClassificationInception(nn.Module):
  def __init__(self, in_channels, out_channels=4):
    super().__init__()

    # 1x1 convolution branch
    self.branch1x1 = nn.Squential(
      nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
      nn.BachNorm2d(out_channels // 4),
      nn.ReLU()
    )
    # 3x3 convolutional branch
    self.branch3x3 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
      nn.BachNorm2d(out_channels // 4),
      nn.ReLU(),
      nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
      nn.BachNorm2d(out_channels // 4),
      nn.ReLU()
    )
    
    # 5x5 convolutional branch
    self.branch5x5 = nn.Sequential(
      nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
      nn.BachNorm2d(out_channels // 4),
      nn.ReLU(),
      nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2),
      nn.BachNorm2d(out_channels // 4),
      nn.ReLU()
    )

    # max pool branch
    self.branch_pool = nn.Squential(
      nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
      nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
      nn.BachNorm2d(out_channels // 4),
      nn.ReLU()
    )

  def forward(self, x):
    out1x1 = self.branch1x1(x)
    out3x3 = self.branch3x3(x)
    out5x5 = self.branch5x5(x)
    out_pool = self.branch_pool(x)
    out = torch.cat((out1x1, out3x3, out5x5, out_pool), dim=1) # 먼저 여기까지 진행한 뒤에 아래의 Test를 진행하여 input의 모습을 확인하고 (정상적으로 출력되는지를 확인할 것) - Batchsize가 변경되었다면 문제가 있는 것
    out = torch.flatten(out, 1)
    out = nn.Linear(out.shape[1], 1)(out)
    out = nn.Sigmoid()(out)
    return out


# Test
input = torch.rand([4, 1, 32, 32]) # [BS, Color, H, W]
model = ImageClassificationInception(in_channels=1, out_channels=128)
model(input).shape
    
    
