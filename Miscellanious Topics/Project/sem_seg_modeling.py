
import openai
import os

openai.api_key = os.getenv('OPENAIAPI')

response = openai.Image.create(
  prompt="요청할 내용",
  n=1, # 만들 그림 숫자
  size="1024x1024" # 만들어질 그림의 해상도
)

image_url = response['data'][0]['url']

image_url

# 그림 편집
# 주의 사항 - 그림은 정사각형이어야 하고, 4MB미만이어야 함
# RGBA 채널을 갖고 있어야 함
# alpha채널이 없다면 아래의 사이트에서 추가해 보는 것도 좋음
# online-tool중 하나 https://onlinepngtools.com/create-transparent-png

response = openai.Image.create_edit(
  image=open('.png', 'rb'),
  mask=open('mask_.png', 'rb'),
  prompt='마스크된 곳에 집어넣을 그림 내용',
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
image_url

response = openai.Image.create_variation(
  image=open('.png', 'rb'),
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
image_url

openai.model.list()

response = openai.Completion.create(
  model='gpt-4o-mini',
  prompt='',
  temperature=0.7,
  max_tokens=64,
  top_p=1.0,
  frequcney_penalty=0.0,
  presence_penalty=0.0
  
)

response['choices'][0]['text']

response = openai.Completion.create(
  model='gpt-4o-mini',
  prompt='',
  temperature=0.7,
  max_tokens=64,
  top_p=1.0,
  frequcney_penalty=0.0,
  presence_penalty=0.0,
  stop=['\n'] # 특정 단어를 생성하게 되면 그 곳에서 멈출 것
)

response['choices'][0]['text']

