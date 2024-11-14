
import pandas as pd
from plotnine import ggplot, aes, geom_text, labs
from sklearn.manifold import TSNE # 차원 감소 기술 중 하나로 차원의 크기를 크게 줄일 수 있음 100차원을 2차원으로 줄일 수 있음
import torchtext.vocab as vocab
import torch

glove_dim=100
glove = vocab.GloVe(name='6B', dim=glove_dim)

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

words = []
categories = ['numbers', 'algebra', 'music', 'science', 'technology']

df_word_cloud = pd.DataFrame({
  'category':[],
  'word':[]
})

for category in categories:
  print(category)
  closest_words = get_closest_words_from_word(word=category, max_n=20)
  temp = pd.DataFrame({
    'category': [category] * len(closest_words)
    'word': closest_words
  })
  df_word_cloud = pd.concat([df_word_cloud, temp], ignore_index=True)


n_rows = df_word_cloud.shape[0]
n_cols = glove_dim
X = torch.empty((n_rows, n_cols))

for i in range(n_rows):
  current_word = df_word_cloud.loc[i, 'word']
  X[i, :] = get_embedding_vector(current_word)
  print(f"{i}: {current_word}")

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X.cpu().numpy())

df_word_cloud['x'] = X_tsne[:, 0]
df_word_cloud['y'] = X_tsne[:, 1]

# 아래에서 word cluster를 graph로 출력!
ggplot(data=df_word_cloud.smaple(25)) + aes(x='x', y='y', label='word', color='category') + geom_text() + labs(title='GloVe Word Embeddings and Categories')
