# batches??
일반적으로 dataset의 크기가 매우 큼 - 한 번에 데이터 셋의 내용을 한 번에 모델로 전달하는 것은 불가능한 일
대신 dataset의 크기를 작게 쪼개어, 모델이 입력으로 조금씩 소모하도록 해야 하는데, 그 묶음 단위의 크기를 만드는 것 - batches

# Batch Size
batch size는 모델에 한 번에 제공되는 batch의 크기 단위
모델의 훈련 속도, 학습 과정의 안정성에 영향을 줌

작은 batch size를 만들어야 할 때
작은 batch size는 데이터의 소음이 많고, 에러를 덜 만들어냄
CPU, GPU의 한계로 메모리에 훈련 데이터를 전달하기 쉬움

일반적인 batch size
1~512
일반적으로 2의 배수를 씀
기본적인 값은 32
