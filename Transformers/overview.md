# RNN의 구조에서 개선이 필요했던 사항
RNN은 sequential로 진행
Natural Language Processing
- 단어는 하나하나 병렬로 처리하기 어려움
- 단어 순서가 중요(영어 기준)
- 커다란 sequence를 가진 데이터를 처리하면 앞의 내용을 잊어버리는 문제가 발생
- 훈련 시키기 어려움 - layer를 통해 전파될 때 gradient 효과가 사라지거나, 발산하거나 하는 등의 문제가 발생
  unrolled-RNN에서 특히 문제가 많았던 것 같음


# Transfomer
google에서 번역을 위해 개발한 모델
주요 장점:
 - 단어 순서를 추적
 - gradient의 사라짐, 발산 효과 없음
 - 훈련이 병렬로 진행 가능
 - 대규모 모델을 가능하게 함

Attention이 주요 특징!

# Transformers는 어떻게 동작하는가?
3가지 특징
positional encoding
Attention
Self-Attention

# positional encoding
Natural Language Processing에서 단어의 순서는 매우 중요함
RNN
 단어가 전달되는 순서대로 단어 순서를 인식 - 병렬로 처리하는 것이 불가능했던 이유
Transformers
 positional encoding
 단어의 순서와 단어를 같이 묶음
 최초의 모델(논문에서 쓰인) sine, cosine을 encoding으로 활용

# Attention
attention 개념 자체는 2015년에 처음 등장
attention은 원 문장의 각 단어들이 번역에서 어떻게 처리되는지를 주목함
단어와 단어 사이의 관계를 훈련동안 배우게 됨
단어 순서, 문법, 맥락 등을 배우는데 도움이 됨

# Self-Attention
맥락 표현에 도움이 됨
예를 들어 같은 단어라도 어디에 있느냐에 따라 뜻이 달라짐 

# 모델들
Google "BERT"
Bidirectional Encoder Representations from Transformers
110M parameters
OpenAI "GPT-3"
Generative Pre-trained transformers 3
175B parameters

Google "LaMDA"
Language Model for Dialogue Applications
137B parameters

# 활용
NLP
  문서 요약, 분류, 맥락 분석
Computer Vision
Time-Series Prediction

