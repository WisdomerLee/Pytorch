# 앞에 #%%로 시작하여 묶게 되면, 해당 파일을 실행할 때 저 앞 단위까지만 진행하고 끊음 > 코드 중간중간의 과정을 전체 실행 없이 중간 중간 확인하고 싶을 때 사용할 것!

#%% package
import torch
import numpy as np
import seaborn as sns


#%% 
x = tensor.tensor(5.5)

y = x + 10
print(y)

print(x.requires_grad)

x = torch.tensor(2.0, requires_grad=True)
print(x.requires_grad)

def y_function(val):
  return (val-3) * (val-6) * (val-4)

x_range = np.linespace(0, 10, 101)
y_range = [y_function(i) for i in x_range]
sns.lineplot(x=x_range, y=y_range)

y = (x-3) * (x-6) * (x-4)
y.backward()
print(x.grad)

x = torch.tensor(1.0, requires_grad=True)
y = x**3
z = 5*x-4
z.backward()

print(x.grad)
x11 = torch.tensor(2.0, requires_grad=True)
x21 = torch.tensor(3.0, requires_grad=True)
x12 = 5*x11-3*x21
x22 = 2*x11**2 + 2*x21
y = 4*x12 + 3*x22
y.backward()
print(x11.grad)
print(x21.grad)
