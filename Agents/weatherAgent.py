from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from dotenv import load_dotenv
import requests

load_dotenv()
@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=62ba0b34f7b7aed696ed5f9492c5bf47&query={city}'

  response = requests.get(url)

  return response.json()
# Setup tool and LLM
search_tool = DuckDuckGoSearchRun()
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2024-02-01",
    temperature=0.7,
    max_tokens=700
)

# Pull prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Create AgentExecutor (FIXED: Removed `prompt`, added `tools`)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
handle_parsing_errors=True
)

# Invoke the agent
response = agent_executor.invoke({"input": "find the weather of datia and ts weather"})
print(response['output'])