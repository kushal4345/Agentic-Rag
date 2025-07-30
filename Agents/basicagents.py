from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

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
    tools=[search_tool],
    prompt=prompt
)

# Create AgentExecutor (FIXED: Removed `prompt`, added `tools`)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

# Invoke the agent
response = agent_executor.invoke({"input": "gimme the three ways to go mumbai"})

print(response['output'])
