# dot product
import numpy as np

X = [0, 1]
w1 = [2, 3]
w2 = [0.4, 1.8]

# 어느 weight가 X와 더 비슷한가?
# dot product로 확인 가능

dot_X_w1 = X[0] * w1[0] + X[1] * w1[1]
dot_X_w1

dot_X_w2 = X[0] * w2[0] + X[1] * w2[1]
dot_X_w2

# numpy 내부에 dot 함수가 있음!!
np.dot(X, w1)

np.dot(X, w2)

