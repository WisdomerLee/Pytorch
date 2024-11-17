
from transformer import pipeline
from sentence_transformers import SentenceTransformer
from scipy import spatial # 가장 비슷한 결과를 계산할 때 사용하는 것??
import pandas as pd


twitter_file = 'data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna().sample(1000, random_state=123).reset_index(drop=True)
df.head()


# 감정 분석
# pipeline에 model을 넘기기
sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
# data 중에서 감정 분석에 쓰일 곳을 찾고
data = df.loc[2, 'text']
# data를 pipeline에 넘기기
sentiment_pipeline(data)

# 비슷한 감정의 Tweet를 찾기
model = SentenceTransformer('sentence-transformers/all_mpnet-base-v2')

df = df.assign(embeddings=df['text'].apply(lambda x: model.encode(x)))

def closest_description(desc):
  data = df.copy()
  inp_vector = model.encode(desc)
  data['similarity'] = data['embeddings'].apply(lambda x: 1-spatial.distance.cosine(x, inp_vector)) # 유사성을 cosine similarity로 계산, embeddings의 거리 계산 방식 중에 하나
  data = data.sort_values('similarity', ascending=False).head(3)
  return data[['text', 'sentiment']]

closest_description('this is amazing')
