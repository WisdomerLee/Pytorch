# 먼저 해당 코드를 작성하기 전에 pip install flask로 flask 패키지 설치 필요

from flask import Flask

app = Flask(__name__)

# 아래와 같이 하면 홈페이지에 바로 접근할 경우 Hello world라는 문자열을 돌려주는 것
@app.route('/')
def home():
  return "Hello world"


if __name__ == "__main__":
  app.run()

# python의 터미널에서 python app.py(실행할 파일의 이름)을 실행하면... > flask로 우리가 정의했던 함수들이 반응함
