Input layer
서로 독립적인 변수들로 구성
batches라는 단위로 묶여 처리
binned data
Categorized data 등을 숫자로 변경하여 입력하여야 함

Dense Layer
각 Input Layer는 Output Layer까지 최종적으로 연결
Fully connected layer로도 부름
대개 non-linear activation function이 적용됨

1D convolutional Layer
Layer가 filter들로 구성되어 있음
Input layer로 들어오는 데이터들 중에 일부 부분들이 전달
Input layer의 모든 노드가 활용

Other Layer Types
Recurrent Neural Networks
recurrent cells를 씀
자체 output을 일정 딜레이를 두고 다시 받아들임
context가 필요한 부분들에 사용

Long short-term memory(LSTM)
"memory cell"을 씀
temporal sequence

Output Layer
              Nodes          Output Layer Activation
Regression - 1             - Linear
Multi-Target Regression - N (target의 숫자) - Linear
Binary Classification - 1 - Sigmoid
Multi-Label Classfication - N (labels의 갯수) - Softmax

