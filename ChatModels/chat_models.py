from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
import os


load_dotenv()


llm = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name (not model name)
    openai_api_version="2025-01-01-preview",
    temperature=0.7
)
while True:
    user_input = input('You :' )
    if user_input == 'exit':
              break
    llm.invoke(user_input)
    print("AI :", llm.invoke([HumanMessage(content=user_input)]).content)
