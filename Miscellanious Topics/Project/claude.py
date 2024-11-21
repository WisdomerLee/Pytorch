# API를 써야 함
# https://console.anthropic.com/settings/keys
# 해당 곳에서 API key를 생성하고 그것을 복사해와야 함
# 그리고 그것을 system의 환경 변수로 설정할 것!

# API가 제대로 동작하는지 확인하는 코드

import anthropic # 이것을 하려면 pip install anthropic으로 패키지를 설치해야 쓸 수 있음
import os

client = anthropic.Anthropic(
  # 기본적으로는 os.environ.get("ANTHROPIC_API_KEY")
  api_key=os.environ.get("CLAUDE_API") # 시스템의 환경 변수에 저장한 환경 변수 이름 - claude key가 들어있어야 함
)

message = client.messages.create(
  model="claude-3-opus-2024-0229",
  max_tokens=1024,
  messages=[
    {"role": "user", "content": "질문할 내용"}
  ]
)
print(message.content)

message.content[0].text # 이렇게 하면 답변의 text만 얻을 수 있음

# 출력을 python의 출력에서 조금 더 예쁘게 확인하고 싶다면
from pprint import pprint

pprint(message.content[0].text)

# console로 가면 얼마나 사용했는지를 확인할수 있음 - API key를 사용하는데 드는 비용이 있음!!!
