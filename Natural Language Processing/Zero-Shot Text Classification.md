# Zero-Shot Classification

# 일반적인 훈련 방식
입력
모델 파라미터
모델 결과

훈련용 데이터 (방대한 문자 데이터)

Neural Network - weights
예측
실제 값과 비교
loss를 계산하고, loss의 값을 토대로 gradient를 계산하고
그것을 다시 weights에 업데이트하는 과정이 필요

그런데 기존의 방식대로 문자의 뜻을 다루려고 한다면... 제한된 클래스 세트에서 모델을 훈련시키면 해당 클래스가 아닌 추가적인 클래스가 발생할 때마다 재 훈련을 처음부터 다시 해야할 수 있음

NLP 영역에서 이를 해결 할 수 있는 방식이 있음

# Zero-Shot Text Classification

훈련용 데이터(방대한 문자 데이터)
Pretrained Network
이미 훈련된 모델을 통해
예측이 나오면

훈련용 데이터에 연관된 Labels들을 훈련 데이터와 별도로 같이 보내기

그러면 후보로 제시된 것들에 어느 쪽에 더 가까운지를 예측하여 그 확률을 계산함

# Zero-Shot의 목표
우리가 원하는 형태로 훈련되지 않은 모델을 활용하고자 할 때
훈련된 모델을 일반화 해서 새롭고, 경험하지 않은 클래스로 만들 수 있음
재 훈련을 방지
새로 추가되는 클래스와 관련된 embeddings와 같은 추가정보가 있음
추가정보는 이미 알려진 클래스와의 관계를 통해 예측
이건 매우 유용한테 종류가 매우 다양할 때, 그리고 그 종류가 자주 바뀔 때 유용 

# Introduction
Attribute-Based, Word Embeddings, Transfer Learning

## Attribute-Based
분류는 기본적으로 속성과 연관되어있음, 예를 들면 동물 같은 것
속성은 날 수 있음, 가죽을 가짐 과 같은 형태가 될 수 있음
훈련: 모델은 클래스와 연관된 속성을 학습
inference : 속성을 결합하여 경험하지 않은 분류를 예측할 수 있게 됨

## Word Embeddings
분류, 객체는 word embedding으로 구성된 연속적인 vector space에 존재
embeddings는 단어와 주제에서 맥락적인 관계를 파악할 수 있음
공간의 지형적인 관계성을 적용할 수 있음

## Transfer Learning
model은 대규모 데이터로 훈련되어있고
'Natural Language Inference'의 기술이 유명 - 자연 언어 추론

# Natural Language Inference
자연 언어 추론은
Premise, Candidate Class, Hypothesis, Prediction의 네 단계를 거침

문자 데이터가 들어오면, 연관된 곳이 어느쪽인지가 제공되고, 이를 통해 모델은 가설을 세움
그리고 그 예측에 해당되는 부분이 어느 클래스에 해당되는지 확률적으로 계산함

다중 분류 레벨, 등에도 활용 가능

# HuggingFace
모델에서 Zero-Shot Classification 모델 분류가 별도로 존재

