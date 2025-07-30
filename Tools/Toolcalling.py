from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
import os 
from dotenv import load_dotenv
load_dotenv()

#creating a tool
@tool
def multiply(a:int,b:int):
    """Given two number tool will return their product"""
    return a*b

print(multiply.invoke(4,3))
print(multiply.name)
print(multiply.description)
print(multiply.args)

#binding a tool

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)

llm_with_tool = llm.bind_tools([multiply])  #tool has been binded llm can use it further
