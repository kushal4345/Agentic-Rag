from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from typing import Annotated
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# LLM Initialization
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)

# Tool 1: Get currency conversion rate
@tool
def get_currency_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the conversion rate between base and target currency."""
    url = f'https://v6.exchangerate-api.com/v6//pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

# Tool 2: Convert value using rate
@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Converts currency using conversion rate."""
    return base_currency_value * conversion_rate

# ✅ Use correct tool name
llm_with_tools = llm.bind_tools([get_currency_factor, convert])

# Step 1: Initial message
messages = [HumanMessage(content="What is the conversion factor between INR and USD, and based on that can you convert 10 INR to USD?")]
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

# Step 2: Handle tool calls
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_currency_factor':
        tool_message1 = get_currency_factor.invoke(tool_call['args'])
        conversion_rate = tool_message1['conversion_rate']  # ✅ use dict access
        messages.append(tool_message1)

    elif tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call['args'])
        messages.append(tool_message2)

# Step 3: Final response from LLM
final_response = llm_with_tools.invoke(messages)
print(final_response.content)
