
information filtering system - 정보 중에 특정 정보만 걸러내기
predicts user preferences and recommends item - 사용자의 선호도를 예측하여 상품을 추천하기

기본 정보, 나이, 성별, 수입 등등

입력으로 쓰이는 것은 기존에 구입한 상품, 검색 등
사용자의 정보로는 기본 정보, 나이, 성별, 수입, ...

인터넷 쇼핑 등에서 쉽게 찾아볼 수 있음

# Content-Based vs Collaborative Filtering

|Recommender Systems|
|--|
|Content-Based|Collaborative Filtering|

추천 시스템은 크게 콘텐츠 기반, 그리고 함께 작동하는 거름망 두 가지 시스템으로 나눌 수 있음

콘텐츠 기반 Content-Based은
사용자가 무엇을 본 것인지 확인하고, 비슷한 것을 추천해주는 것
Collaborative Filtering은 사용자에 초점을 맞춰 비슷한 사용자가 확인한 것을 이용하여 
비슷한 기존 사용자가 이미 본 것이 있으면, 해당 내용을 비슷한 것을 본 다른 사람에게 추천

Collaborative Filtering은 Item CF, User CF, Matrix Factorization으로 구분

# Matrix Factorization
Interaction Matrix = User Matrix * Item Matrix

# Neural Collaborative Filtering Network

User-Input, Item-Input
각각 Embedding으로 변환
Concatenated Tensor
Fully-Connected
Output

# Evaluation Metrics

Precision@k = $#relevant recommendations \over #recommended items$
연관된 아이템들을 추천할 수 있는 능력을 측정

Recall@k $#relevant recommendations \over # all possible relevant items$
연관성 없는 아이템을 거부할 수 있는 능력을 측정

# Data
코드 구현에서 사용하게 될 데이터
MovieLens Dataset
사용자에게 영화 추천하는 데이터
작은 데이터 셋이 사용
10만개 평가
3600개의 태그
9000개의 영화
600명의 사용자
2018년 9월까지의 데이터
https://grouplens.org/datasets/movielens/
