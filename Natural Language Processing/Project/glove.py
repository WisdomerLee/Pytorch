# 아래의 것을 실행하기 위해서는 pip install torchtext 필요
import torch
import torchtext.vocab as vocab

# nlp와 관련된 내용은 다음의 링크를 참고하면 좋음
# nlp.stanford.edu/projects/glove/

# 아래의 코드를 실행하면 .vector_cache라는 폴더에 해당 모델의 파라미터가 담긴 파일이 생성
glove = vocab.GloVe(name='6B', dim=100)

# 이미 들어있는 단어들의 숫자와 embeddings의 차원 숫자를 확인해보기
glove.vectors.shape

# glove.stoi['단어'] 로 해당 단어의 index를 알 수 있고
# glove.itos[숫자]로 해당 index에 있는 단어를 알 수 있음

# 단어의 embedding vector를 얻어보기
def get_embedding_vector(word):
  word_index = glove.stoi[word]
  emb = glove.vectors[word_index]
  return emb

get_embedding_vector('chess').shape

# 가장 가까운 단어를 찾기!
def get_closest_words_from_word(word, max_n=5):
  word_emb = get_embedding_vector(word)
  distances = [ (w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos] # embedding vector를 cpu로 갖고와서 item으로 얻을 것, # embeddings의 vector간 거리로 가장 가까운 거리의 단어들을 찾기
  dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n] # 가장 가까운 순서대로 정렬
  return dist_sort_filt

get_closest_words_from_word('chess')

# embeddings에서 가장 가까운 단어 찾기
def get_closest_words_from_embedding(word_emb, max_n=5):
  distances = [ (w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos] # embedding vector를 cpu로 갖고와서 item으로 얻을 것, # embeddings의 vector간 거리로 가장 가까운 거리의 단어들을 찾기
  dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n] # 가장 가까운 순서대로 정렬
  return dist_sort_filt

# 유추를 활용하기

def get_word_analogy(word1, word2, word3, max_n=5):
  # logic_w1=,..
  # w1 - w2 + w3 -> w4
  word1_emb = get_embedding_vector(word1)
  word2_emb = get_embedding_vector(word2)
  word3_emb = get_embedding_vector(word3)
  word4_emb = word1_emb - word2_emb + word3_emb
  analogy = get_cloest_words_from_embedding(word4_emb)
  return analogy

get_word_analogy(word1='sister', word2='brother', word3='nephew')
