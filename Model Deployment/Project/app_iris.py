# 먼저 해당 코드를 작성하기 전에 pip install flask로 flask 패키지 설치 필요

from flask import Flask
from model_class import MultiClassNet # 모델을 불러오기 위해서 모델의 클래스를 불러오기
import torch
import json

# 모델 객체 생성하기
model = MultiClassNet(HIDDEN_FEATURES=6, NUM_CLASSES=3, NUM_FEATURES=4)
local_file_path = 'model_iris.pt'
model.load_state_dict(torch.load(local_file_path)) # 저장된 모델의 학습 파라미터를 가져오기!


app = Flask(__name__)

# 아래와 같이 하면 홈페이지에 바로 접근할 경우 Hello world라는 문자열을 돌려주는 것
@app.route('/')
def home():
  return "Hello world"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  if request.mothod == 'GET':
    return 'Please use the Post method'
  if request.method == 'POST':
    print(request.data.decode('utf-8'))
    data = request.data.decode('ut-8')
    dict_data = json.load(data.replace("'", "\"")) # json은 데이터에서 반드시 더블따옴표를 사용하여 키, 값을 지정, 구분하기 때문에 python의 경우엔 꼭 필요한 처리
    
    # 하기 전에 가짜 데이터로 모델이 동작하는지부터 확인해보기
    # X = torch.tensor([[5.1, 4.3, 2.3, 1.8]])
    # 전달된 데이터로 확인하기!
    X = torch.tensor([dict_data["data"]])
    y_test_hat_softmax = model(X)
    y_test_hat = torch.max(y_test_hat_softmax, 1)
    y_test_cls = y_test_hat.indices.cpu().detach().numpy()[0]
    cls_dict = {
      0: 'setosa',
      1: 'versicolor',
      2: 'virginica'
    }
    return f"Your flower belongs to class {cls_dict[y_test_cls]}"

if __name__ == "__main__":
  app.run()

# python의 터미널에서 python app.py(실행할 파일의 이름)을 실행하면... > flask로 우리가 정의했던 함수들이 반응함
