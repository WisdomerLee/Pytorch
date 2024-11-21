# Introduction
특정 타입을 가진 것을 모은 database
저장, 관리, 데이터 추출을 지형적인 방식으로 ...?
NLP 같은데서 많이 활용하는 database
similarity search
clustering
real-time analytics

위의 세 기능 동작

# Vector Databases가 필요한 이유
일반적인 인터넷 데이터 - 그림, 글자, 영상, 소리 등 특정 구조 없는 data
일반적인 데이터베이스 - SQL  구조화된 data

# Image Similarity
가장 닮은 그림을 찾는다면??

# Text Similarity
가장 비슷한 내용을 담은 글은?

# Embeddings
글자 -> NLP -> word embeddings
그림 -> CNN, ViT -> image embeddings

# 그림 embeddings
그림이 들어오면..
일반적으로 CNN 혹은 ViT 모델에서 
가장 마지막 층에서 클래스 분류를 진행하는데, embedding으로 변환하는 것은 해당 모델의 마지막 레이어를 제외한 나머지를 이용하여
해당 모델의 출력을 활용
훈련된 모델을 그대로 활용하여 쓰기 때문에 별도의 훈련과정이 필요 없는 부분은 장점

# 가장 비슷한 데이터 찾기
가장 비슷한 것을 어떻게 정의하느냐에 따라 가장 비슷한 데이터가 달라짐
간단한 것은 embeddings가 있는 embeddings 공간에서 해당 데이터간 거리를 측정하여 가장 짧은 거리를 가진 데이터를 찾을 수 있음

# 저장된 embeddings
작은 데이터의 경우 np.array()로 배열로 불러올 수 있으나..
매우 많은 데이터가 저장된 경우 해당 내용은 불가능할 뿐더러, 몹시 느림

# Indexing
어떤 그림, 글자 데이터 이름이 있고, 해당 embedding이 있고, 각 embedding에 index를 할당
index는 검색 속도를 높여주는 효과를 갖고 있음

# 이용 가능한 것들
pinecone.io
trychroma.com
redis.com
대표적으로 위의 세 가지가 있음

