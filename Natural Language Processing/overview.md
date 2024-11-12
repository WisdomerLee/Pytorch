# 자연어 처리
컴퓨터가 사람이 쓰는 말을 이해하고, 해석하고, 생성할 수 있도록 진행하는 영역
Deep Learning에서는 글, 소리를 분석, 생성하는 쪽과 연관되어 있음
정제되지 않은 말에서 뜻을 추출해낼 수 있음

Chatbot, 감정 분석, 번역, 음성 인식

# NLP model이 하는 일
글자가 들어오면 Neural Network에서
1. 다음에 오게 될 말을 확률적으로 계산하여 맞는 단어들을 순서대로 계산
2. 감정을 분석
3. 요약
4. 번역을 할 수 있음

# Neural Network는 숫자만 입력으로 받음
글자로 들어간 것을 숫자로 변경할 필요가 있음!
글자를 숫자의 표현형태로 변경해야 함

Word Embedding
대개 기본으로 사용하는 Neural Network에 따라 달라짐

# 알아야 할 단어
Tokenization > Word Embedding으로 변환하기 위해 필요한 과정
긴 글을 작은 단위로 쪼개는 과정, 단어로 쪼갤 수도, 그보다 더 작은 단위로 쪼갤 수도 있음

Document - token화된 것을 묶은 것
Document의 묶음 - Corpus
