# AI 모델을 어디에 ??
AI 모델을 잘 만들고, 훈련시키고 했다면 해당 모델을 활용할 수 있어야 함
그런데 이것을 어디에 둘 것인가는 큰 문제 중 하나

On-Premise, Cloud
On-Premise의 경우 모델의 연산을 감당할 수 있는 컴퓨터가 있을 경우..

Cloud
일반적으로 AI의 연산을 감당할 수 있는 컴퓨터는 매우 비싼 편에 속하기 때문에 그 기기를 갖추지 못했을 경우 선택할 수 있음
크게 3가지 경우가 있음
AWS, Azure, GCP

# 모델을 어떻게 활용할 수 있는가?
일반적인 API  기준은  REST API (REpresentational State Transfer, Application Programming Interface)
웹서비스는 API를 통해 사용자의 요구를 받고 처리한 뒤 돌려줌
API는 어플리케이션들이 서로 어떻게 의사소통을 하는지를 정의한 인터페이스

또 다른 방식은 GraphQL, gRPC라는 방식도 있다고 함...

# URL의 구성
Protocol / domain name port application resouce parameter
위와 같은 구성으로 되어있음
http://~.com:800/we/f/{}
위와 같은 경우 protocol을 http를 쓸 거고, ~.com의 도메인에 접속하며, 그 접속경로는 800이고, application으로 we를 써서, 그 중에 f에 해당되는 것을 {}의 파라미터를 주고 이용할 것

이 중에 RESTful endpoint에 해당되는 부분은 http://~.com:800그 이후에 해당되는 부분들
port 번호가 생략되어있는 경우는 기본 값을 쓰는 경우 - 대체로 80을 사용함

# HTTP-Methods
GET, POST 요청이 들어오면 data가 server로 전달되고, 그 값은 JSON 형태로 전달

GET은 데이터를 받아올 수 있고
POST는 서버에 데이터를 추가할 때 사용함 - 그렇게 약속한 것
DELETE는 서버에 있는 데이터를 지울 때
PUT은 데이터를 업데이트 할 때 사용함

저 중에 가장 많이 쓰는 것은 GET, POST

# Flask
python에서 제공하는 web framework
가볍고 유연하고 쓰기 간편

# API 테스트
Postman이라고 하는 프로그램으로 API의 기능이 동작하는지 확인할 수 있음
사용자가 보기 편한 인터페이스를 제공하기 때문에 사용법도 간편


