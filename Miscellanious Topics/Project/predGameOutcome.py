# crewAI에 대한 예시들도 많이 있음!
# github.com/joaomdmoura/crewAI-examples를 참고해보기



import os
from IPython.display import Markdown
from crewai import Agent, Task, Crew  # 해당 내용을 쓰려면 pip install crewai를 실행하여야 함!
from crewai_tools import ScrapeWebsiteTool, SerperDevTool # 해당 내용을 쓰려면 역시 pip install crewai-tools 로 패키지 설치가 선행되어야 함
from langchain_groq import ChatGroq # pip install langchain-groq으로 패키지 설치 필요
from dotenv import load_dotenv
from pydantic import BaseModel

# 환경 변수로 설정해둔 API key들이 저장된 .env 파일 불러오기
load_dotenv()

# 도구 설정
search_tool = SerperDevTool() # 인터넷 검색 도구
scrape_tool = ScrapeWebsiteTool() # 검색된 페이지에서 내용 추출 도구

# 사용할 모델 이름 설정
MODEL = "mixtral-8x7b-32768" # 모델을 선택할 때는 모델이 동작하는데 필요한 사양(파라미터에 비례)과 정확도를 같이 고려할 것

# LLM 설정, langchain을 통해 ChatGroq에 접근하는 것은 모델에 대한 접근을 제공하는 서비스를 이용하는 것에 가까움, 쉽게 말해 구독 같은 것
llm = ChatGroq(
  temperature=0,
  model_name=MODEL,
  api_key=os.environ["GROQ_API"]
)

# agent 지정 - role로 역할 담당 이름을 부여하고, goal로 목표를 설정하고, backstory로 세부적인 내용을 지정
match_researcher = Agent(
  role="Match Researcher",
  goal="Find the match result history of previous matches between
  {country1} and {country2} in men's soccer. Consider also the
  recent trends and statistics of the countries' performance.",
  backstory="""
  You are a researcher who specializes in finding historical
  information about previous matches between two countries in
  men's soccer. You consider direct matches, but also the recent
  performance of the teams. Your work is the basis for Match
  Result Predictor to predict the outcome of the match.
  """,
  allow_delegation=False,
  llm=llm,
  max_iter = 2,
  tools=[search_tool, scrape_tool],
  verbose=True
)

match_predictor = Agent(
  role="Match Result Predictor",
  goal="Predict the outcome of the match between {country1} and
  {country2} in men's soccer.",
  backstory="""You are a match predictor who predicts the outcome
  of the match between two countries. You base your prediction on
  the results of previous matches and recent trends and
  statistics of the countries' performance """,
  allow_delegation=False,
  context=["researcher_direct_match", "researcher_recent_trend"]
  llm=llm,
  max_iter = 2,
  verbose=True
)

# agent가 수행할 임무들 지정
analyze_matches = Task(
  description=(
    "1. Find historical soccer results of previous matches between {country1} and {country2}"
  ),
  expected_output="A list of previous matches between the two countries.",
  agent=match_researcher,
)

class OutputFormat(BaseModel):
  country1: str
  country2: str
  prediction: str

predict_match_outcome = Task(
  description=(
    "Predict the outcome of the match between {country1} and {country2} based on direct match results and recent trends and statistics.\n"
    "Only provide one most likely result.\n"
  ),
  expected_output="A markdown document on the teams and the match result prediction.".
  agent=match_predictor,
  output_format="markdown",
  output_format_model=OutputFormat,
  output_format_description=(
    "The output format is a markfown file with the following structure:\n"
    "1. {country1}\n"
    "2. {country2}\n"
    "3. Match Prediction\n"
  ),
  output_file="match_prediction.md"
)

# 위에서 설정한 agent, task를 아래에 넣고
crew = Crew(
  agents=[match_researcher, match_predictor],
  tasks=[analyze_matches, predict_match_outcome],
  verbose=2
)

# 그 내용을 실행할 것!
result = crew.kickoff(inputs={"country1": "Germany", "country2": "Scotland"})

Markdown(result)

