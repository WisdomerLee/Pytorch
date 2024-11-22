# Introduction
Agent는 무엇? LLM을 이용한 어플리케이션은 특정 업무를 수행
주요 요소는 agent

사용자의 요구에 맞게 계획을 세우고 planning, 저장 memory 하고, 도구 tools를 이용함
동작을 수행할 수 있음

# Used framework
crewai
사용하기 간단, agents들의 상호작용을 간단히 할 수 있음
특징이 많음

# AI Crew는 무엇?
여러 agent가 협업하여 동작을 수행하는 것
AI Crew 내부에 특정 분야의 전문 Agent들이 있음

# 예시 - 방학 계획 세우기
목표를 정의
사용자의 입력을 받음
설정
agents, tasks, tools process
사용자는 시작 도시, 이동할 수 있는 거리, 접근할 수 있는 도시, 취미 등을 입력하면
AI Crew에서 Agents들이 도시를 고르거나, 지역 전문가 혹은 취미 등을 알선해주는 역할을 나누고
그 역할에 맞게 목적을 설정하고, 할당된 도구를 사용

# Agents 예시
AI Crew에서 도시 고르기, 지역 전문가, 예약 전문가로 나누어
역할을 분배하고, 각각의 목적을 부여함, 그 전문가의 배경도 설정 

# Tasks 예시
장소 식별, 정보 얻기, 계획세우기 등으로 잘게 쪼개진 업무들
여행 세우기 -> 장소 식별 -> 해당 장소의 정보 얻기 -> 얻은 정보로 계획 세우기 등으로 단계별로 나누거나 하는 것

# Tools
Agents들이 사용할 수 있는 도구들 인터넷을 통한 검색이나, webpage에서 가져오거나, 특정 파일의 내용을 불러오거나 등등

# Memory
## Short-Term Memory
일시적으로 상호작용을 저장하는 메모리, agent가 현재 맥락에 맞는 정보를 가져올 수 있음

## Entity Memory
정보를 파악하고 구성
entities, 사람, 정보 등이 포함됨

## Long-Term Memory
통찰, 결과 등의 내용, agent가 시간이 지나면서 축적하는 지식

## Contextual Memory
상호작용간의 맥락을 파악할 수 있음
agent의 답변에 따라 점차 연관성이 증가

# Memory 구현
구현은 상대적으로 간단
기본적으로는 메모리 상태가 꺼진 상태, openai embeddings

```
from crewai import Crew, Agent, Task, Process

crew = Crew(
  agents=[],
  tasks=[],
  process=Process.sequential,
  memory=True,
  verbose=True
)

```
# Memory 활용의 장점
추가적인 학습이 가능 - crew들은 새 정보를 받아들이고, 임무를 달성하는 방식을 다시 설정할 수 있음
개인화된 부분에 맞춤 가능 - agents 들이 사용자 정보를 기억하고, 그 전의 대화 내용을 기억함
성능의 개선 - 더 나은 정보로 결정이 가능하고, 이전의 지식으로 맥락적인 통찰이 가능

# Asynchronous Operation
모든 프로세스가 순차적으로 진행될 필요가 없음
이전의 결과나 그런 것에 영향을 받지 않는다면 프로세스를 평행하게 진행할 필요가 있음

# Callbacks
task callback, step callback
특정 단계나 해당 과정이 끝날 경우 호출되는 함수를 지정할 수 있음
대개 어떤 진행 상태를 알거나, 다른 행동에 활용될 수 있음
task 내부의 파라미터로 전달

# Collaboration
Agents는 임무를 함께 수행하여
정보를 공유, 특정 임무를 보조, 리소스 최적화
LLM 관리

```
from crewai import Agent, Task, Crew, Process
crew = Crew(
  agents=[planner, writer, editor],
  tasks=[plan, write, edit],
  verbose=2,
  manager_llm=llm,
  process=Process.hierarchical # 이렇게 하면 병렬로 수행 가능한 부분은 병렬로 처리!
)
```

# 예상되는 임무 결과
output format은 사용자가 정의할 수 있음!
```
class OutputFormat(BaseModel):
  chapter_title: str
  bullet_points: list[str]

Task(
  description=("업무의 내용, 목표 등"),
  expected_output="출력될 형태의 데이터",
  agent=editor,
  output_format="markdown",
  output_format_model=OutputFormat,
  output_format_description=("output_format_model에 대한 간략한 설명"),
  output_file = ".md"

)
```
