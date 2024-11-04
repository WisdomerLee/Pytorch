#
import pandas as pd
from sklearn import model_selection, preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error

# 데이터 불러오기
df = pd.read_csv('ratings.csv')
df.head(2)
print(f"Unique Users: {df.userId.nunique()}, Unique Movies: {df.moveId.nunique()}")

# 데이터 클래스
class MovieDataset(Dataset):
  def __init__(self, users, movies, ratings):
    super().__init__()
    self.users = users
    self.movies = movies
    self.ratings = ratings
  
  def __len__(self):
    return len(self.users)
    
  def __getitem(self, idx):
    users = self.users[idx]
    movies = self.movis[idx]
    ratings = self.ratings[idx]
    return torch.tensor(users, dtype=torch.long), torch.tensor(movies, dtype=torch.long), torch.tensor(ratings, dtype=torch.long)
    
    
# 모델 클래스
# 의외로 추천 시스템의 neural network는 간단한 편...
class RecSysModel(nn.Module):
  def __init__(self, n_users, n_movies, n_embeddings=32):
    super().__init__()
    
    self.user_embed = nn.Embedding(n_users, n_embeddings)
    self.movie_embed = nn.Embedding(n_movies, n_embeddings)
    self.out = nn.Linear(n_embeddings * 2, 1)

  def forward(self, users, movies):
    user_embeds = self.user_embed(users)
    movie_embeds = self.movie_embed(movies)
    x = torch.cat([user_embeds, movie_embeds], dim=1)
    x = self.out(x)
    return x
    
# 사용자, 영화 id를 0부터 시작하도록 
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df['userId'] = lbl_user.fit_transform(df['userId'])
df['moviId'] = lbl_movie.fit_transform(df['movieId'])
df

# 훈련, 평가 데이터 나누기
df_train, df_test = model_selection.train_test_split(df, test_size=0.2, random_state=123)
# Dataset 객체 만들기
train_dataset = MovieDataset(
  users = df_train.userId.values,
  movies = df_train.movieId.values,
  ratings =df_train.rating.values
)
valid_dataset = MovieDataset(
  users = df_test.userId,values,
  movies = df_test.movieId.values,
  ratings = df_test.rating.values
)

# Data Loader
BATCH_SIZE = 4
train_loader = DataLoader(dataset = train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)
test_loader = DataLoader(dataset = valid_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True
)
# 모델 객체 생성, loss function, optimizer
model = RecSysModel(
  n_users = len(lbl_user.classes_),
  n_movies = len(lbl_movie.classes_)
)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()

# 훈련
NUM_EPOCHS = 1

model.train()
for epoch_i in rnage(NUM_EPOCHS):
  for users, movies, ratings in train_loader:
    optimizer.zero_grad()
    y_pred = model(users, movies)
    y_true = ratings.unsqueeze(dim=1).to(torch.float32) # 차원을 맞추고, 데이터 형태를 같게 만들어서 비교 가능하게 만들어야 함
    loss = criterion(y_pred, y_true)
    loss.backward()
    optimizer.step()

# 평가
y_preds = []
y_trues = []

model.eval()
with torch.no_grad():
  for users, movies, ratings in test_loader:
    y_true = ratings.detach().numpy().tolist()
    y_pred = model(users, movies).squeeze().detach().numpy().tolist()
    y_trues.append(y_true)
    y_preds.append(y_pred)

mse = mean_square_error(y_trues, y_preds)
print(f"Mean Squared Error: {mse}")

# 사용자, 영화 목록
user_movie_test = defaultdict(list)

# Precision and Recall
with torch.no_grad():
  for users, movies, ratings in test_loader:
    y_pred = model(users, movies)
    for i in range(len(users)):
      user_id = users[i].item()
      movie_id = movies[i].item()
      print(y_pred) # 먼저 찍어보고 아래처럼 차원을 줄여서 실제 값을 뽑아야 함
      pred_rating = y_pred[i][0].item()
      true_rating = ratings[i].item()
      print(f"User: {user_id}, Movie: {movie_id}, Pred:{pred_rating}, True: {true_rating}")
      user_movie_test[user_id].append(pred_rating, true_rating)

# 위와 같이 하면 모든 아이템의 연관성만 나올 뿐, 사용자가 원하는 n개의 아이템을 찾으려면...
# 그래서 아래와 같이 연관성이 가장 높은 k 개만 찾도록 수정

# Precision@k, Recall@k
predictions = {}
recalls = {}

k = 10 # 연관성이 높은 10개만
thres = 3.5 # 또한 영화 평점이 높은 것들로 추려서

for uid, user_ratings in user_movie_test.items():
  # user ratings by ratings
  user_ratings.sort(key= lambda x: x[0], reverse=True)

  # 연관된 아이템들 숫자
  n_rel = sum((rating_true >= thres) for (_, rating_true) in user_ratings)
  # 추천된 아이템 숫자 - 해당 아이템은 연관성이 높고, k번째 안에 들어감
  n_rec_k = sum((rating_pred >= thres) for (rating_pred, _) in user_ratings[:k])
  # 추천되고, 연관된 아이템 숫자
  n_rel_and_rec_k = sum((rating_true >= thres) and (rating_pred >= thres) for (rating_pred, rating_true) in user_ratings[:k])

  print(f"uid: {uid}, n_rel: {n_rel}, n_rec_k: {n_rec_k}, n_rel_and_rec_k: {n_rel_and_rec_k}")
  precision[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
  recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

print(f"Precision@{k}: {sum(precisions.values()) / len(precision)}")
print(f"Recall@{k}: {sum(recalls.values()) / len(recalls)}")
