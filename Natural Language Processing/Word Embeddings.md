# Word Embeddings
무엇인가?
단어를 숫자로 변경
모든 단어는 고차원의 공간의 tensor로 변환
다른 단어와의 관계는 tensor에서 거리 등으로 확인할 수 있음
이론적으로 비슷한 단어는 거리가 가까움
Deep Learning은 embeddings를 적용
embeddings는 뜻을 표현

# 단어에서 Tensors로
입력된 문장이 있으면 - tokenization으로 분리
각 token화된 것들이 tensor로 변환
문장은 그 tensor들의 list

# Word Embedding Approachs
변환하는 방법의 접근 방식
One-Hot Encoding
Frequency-Based
Neural Network

# One-Hot Encoding
인코딩을 한 번에
매우 간단한 방식
전달되는 순서대로 단어가 index를 가짐
index, 단어를 합친 matrix가 생성

# One-Hot Encoding의 문제
dimensionality -> 메모리 issue발생
Matrix가 비어있는 부분이 많게 됨, 지나치게 쓸모없이 차원만 거대해짐
단어가 각각 분리되어있게 됨
모든 단어가 각각의 단어와 같은 거리를 갖게 됨

# Frequency-Based

## Count
One-Hot Encoding의 방식과 비슷
단어가 얼마나 자주 등장하는지를 판별하는 Count라는 값이 존재
document에서 단어가 얼마나 등장하는지를 세어 Count에 할당

## TF-IDF
Term-Frequency/Inverse Term Frequency
document와 corpus에서 등장하는 단어의 숫자를 셈
document에서 단어가 자주 등장하면 -> 중요한 단어
corpus에서 자주 나오는 단어 -> 중요하지 않은 단어 (뜻 없이 사용되는 단어들 예를 들면 영어의 The, A 같은 것들)

## Co-Occurrence
단어들의 유사성을 찾음
