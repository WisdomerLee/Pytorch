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

# 클라우드에서 REST API로 활용할 수 있게 하는 방법!

google cloud의 console로 들어가 검색으로 function(함수)를 검색 그러면 검색 결과로 나오는 cloud functions를 고르기

함수를 만들 수 있는데, create function을 골라서 클라우드에서 돌아가는 함수를 만들어보기
environment로 버전에 따른 관리를 별도로 할 수 있으며
function name을 지정하고, region을 골라 서비스 되는 지역을 선택할 수 있음

HTTP를 사용하게 되면 http에서 활용할 수 있는 http url이 자동으로 생성됨
또한 해당 내용을 이용할 수 있는 사용자를 로그인이 된 특정 사용자들 혹은 그런 것 없이 접근 가능하게 할 수 있음

또한 모델의 크기에 따라 실행될 메모리 크기 조건을 꼭 설정할 것 > 그렇지 않으면 ... 파괴적인 반응시간을 볼 수 있음...
다 설정하면 저장할 것

그리고 나면 클라우드에서 동작하는 코드를 설정할 수 있는데, 우리는 코드를 파이썬으로 작성해서 동작시킬 것이므로, 코드동작을 파이썬으로 바꾸어줄 것
requirements.txt 파일에는 해당 코드가 동작하기 위해 필요한 라이브러리 등을 저장해둘 것

## 구글 클라우드 함수를 이용하여 사용자의 접근 권한을 지정할 수 있음
principal을 정하여 어떤 사용자가 해당 함수에 접근할 수 있는지 지정할 수 있는데, 클라우드 함수를 호출하는 모든 경우로 지정할 경우
Assign roles에 Cloud Functions에 Cloud Functions Invoker를 지정하면 해당 함수를 호출하는 모두에게 권한을 줄 수 있음
