# Graph는 무엇인가?

Graph - data structure
node, edge로 구성
node들은 edge로 연결되어 있고,
edge들의 구성에 따라 방향성이 있기도(directed), 없기도 함(undirected)

# Graph Formats - Dense Matrix
node가 서로 어떻게 연결되어있는지를 dense matrix로 표시할 수 있음

# Dense and Sparse Matrix

Sparse Matrix (COO) 
source, destination으로 나뉘어 
차원 : edge의 숫자들이 쓰인 2개의 vector 

Dense Matrix
source, destination의 조합으로 구성된 matrix
차원: nodes x nodes -> 노드 숫자의 제곱

Dense Matrix는 node의 모든 조합을 바로 볼 수 있으나, 차원이 큰 단점이 있는 반면
Sparse Matrix는 node의 모든 조합을 보다 낮은 차원으로 확인할 수 있다

# Graph Neural Networks는 무엇??
Neural Network Architecture - Neural Network 구조
deep learning과 graph 이론이 결합
Graphs에 적합하고
graphs를 분석하는데 도움이 되고
오브젝트 간 관계를 이해할 수 있음
예측이 가능

# GNN은 어떻게 동작하는가?
input graph - input data type이 graph

target node에 해당되는 것이 들어가 Graph Neural Network로 학습
target node에서 다른 node로 연결되는 곳을 찾음, 그 다음 node가 연결되는 곳을 찾음

이 과정이 입력 graph 전체를 반복

# GNN, CNN
CNN와 GNN은 많은 부분에서 비슷한 부분이 있음
CNN은 Computer vision에서 매우 유용하나 (euclidean space)
Graph 관련해서는 (non-euclidean space)에는 맞지 않음

# GNN의 활용
Link Prediction
---

node들간의 연결을 예측
SNS의 유저들이 서로 알만한 관계인지를 추천하는 것에 활용될 수 있음!


Graph Classification
---

그래프 클래스 예측
예를 들면 글자 분류, social network 분석


Node Classification
---

node 클래스 예측
전체 graph는 알 수 없지만 해당 node가 어떤 유형에 속하는지를 예측하는 것

# 장단점
장점
nodes/edges의 표현을 배우고, 복잡한 관계를 파악할 수 있음
다양한 크기의 그래프 구조를 가진 것들을 예측할 수 있음
데이터의 크기에서 보다 유연하게 처리할 수 있음
다양한 업무에서 활용할 수 있음 node classification, link prediction, graph classification, ...
실 생활 즉 social network, 추천 시스템, 약 발견 등에 활용 가능

단점
대체로 network 깊이가 얕아, 매우 커다란 dataset에는 적합하지 않음
graph 구조가 바뀌면 재학습을 자주 시켜야 함
연산 비용이 많이 듦
특정한, 정제된 훈련 데이터가 필요
데이터에 잡티가 많거나, 적을 경우 과적합이 발생할 수 있음
아직 새로 나온지 얼마 되지 않아 표준화 작업이 더 필요, 구조라거나 metrics등

# 코드에서 사용하게 될 것
PubMed dataset 사용
19,717개의 과학 논문으로 구성
당뇨병 유형을 3개로 분류
500개의 feature
88.648 edges
node class를 예측

