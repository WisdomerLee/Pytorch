# 먼저 해당 코드를 작성하기 전에 pip install flask로 flask 패키지 설치 필요
# 구글 클라우드로 가서 console 로 들어가기
# google cloud의 storage로 들어가서 - 들어가면 먼저 프로젝트를 만들어야 함
# 프로젝트를 만든 뒤에 bucket을 만들어서
# 이름을 짓고, bucket을 만들면 모델의 파라미터 저장된 부분을 해당 storage쪽에 올릴 수 있음
# 프로젝트의 bucket의 storage에 파일을 올렸다면
# 해당 파일에 대한 접근 권한을 별도로 설정해주어야 함 아무나 사용할 수 있게 한다면...? > 나중에 문제가 발생할 수 있으므로 반드시!!
# 모든 파일에 대한 접근 권한이 아닌, 특정한 파일에만 접근할 수 있도록 수정할 것 > 
# bucket에서 object의 내용으로 가서 파일을 누르면, 파일의 접근권한을 별도로 설정할 수 있는데
# 우리는 프로젝트로 접근할 때 별도의 사용자 인증 없이 접근하게 한다면 > entity1에 public, name1에 allUsers, access1에 reader로 설정하기!
# 읽기 전용으로 접근 권한을 두게 해야 함!
# 그렇게 저장하면 public access에 해당하는 url이 생성되는데 해당 url을 이용하여 해당 모델의 파라미터를 다운 받아 사용할 수 있도록 설정하면 됨!



from flask import Flask
from model_class import MultiClassNet # 모델을 불러오기 위해서 모델의 클래스를 불러오기
import torch
import json
import requests

# 모델 파라미터 다운받기!
URL = '' # 여기에 아까 위에서 설명한 bucket에 있던 url을 여기에 넣어야 함!!
r = requests.get(URL)
local_file_path = 'model_iris_from_gcp.pt'

with open(local_file_path, 'wb') as f:
  f.write(r.content)
  f.close()


# 모델 객체 생성하기
model = MultiClassNet(HIDDEN_FEATURES=6, NUM_CLASSES=3, NUM_FEATURES=4)
# local_file_path = 'model_iris.pt' # 이제 로컬이 아닌 클라우드에서 weights를 받아서 사용할 것이므로...!

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
