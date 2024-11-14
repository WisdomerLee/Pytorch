# 확인
Word Embeddings : 단어를 낮은 차원의 벡터로 수학적인 공간에서 그 뜻과 맥락을 표현할 수 있는 것

# Word Embedding 접근

Neural Network로 하는 방식
Neural Network마다 word를 바꾸는 embeddings의 차원은 모두 다름

Word2Vec - Continous Bag of Words, Skip Gram
GloVe
Bert
GPT

# Neural Network기반 embeddings
목적은
맥락, 뜻 확인
다른 단어들과의 유사성 확인
차원 줄이기
메모리 이슈 피하기

Corpus(Document 묶음)에서 Neural Network를 통해 Word Embeddings 생성

# Word2Vec: Continous Bag of Words
2013년 구글에서 개발한 방식
독립적인 특성, 연관있는 특성을 분리
해당 단어를 중심으로 독립적인 부분으로 좌우를 나눔??
sliding window가 있어서, 전체 텍스트와 모든 단어를 위치마다 살펴보는 것

Input이 들어오면
word단위로 Embedding이 진행되고
모두를 합쳐 Averaging - Linear - Softmax - 예측 - loss로 훈련

# Word2Vec: Skip Gram
기본적으로 Continous Bag of Words와 같은 방식으로 접근
입력이 들어오면
embedding을 거쳐 Averaging - Linear - Softmax - output으로 독립된 특성을 출력!

Continous Bag of Words와 Skip Gram에서 활용하는 모델은 기본적으로 같은 구조를 가지고 있음

# GloVe

단어를 표현하는 Global Vectors
2014년의 논문 Jeffrey Pennington, Richard Socher, Christopher D. Manning의 2014년 GloVe: Global Vectors for Word Representation
Corpus내의 단어의 co-occurrence matrix를 기반으로 같은 맥락에서 단어가 얼마나 자주 등장했는가를 셈
matrix를 만들고, 이 matrix를 기반으로 word embedding 진행
singular value decomposition(SVD)기반으로 만들기
결과 embeddings는 dense, low-dimensional vector
word를 다른 단어들의 vector로 encoding

# BERT
Bidirectional Encoder Representations from Transformers
2018년 구글에서 개발
Pre-trained word embedding
Transformers 기반
'masked language modeling'적용 - 문장에서 몇 단어를 가리고 그것을 예측하는 형태로 학습
다음 단어 예측'next sentence prediction'을 적용 두 문장간의 유사성을 학습

기본형: BERT-base(110m parameters, 440MB), BERT-large(340m parameters, 1.3GB)
다른 형태: RoBERTa, ALBERT, ELECTRA,

# GPT
Generative Pre-trained Transformers
OpenAI에서 개발
word embedding인데 조금더 맥락을 품고 있는 단어 embedding
주변에 있는 단어들에 따라 embedding이 결정
transformer architecture 적용
GPT-3: 175B parameters

