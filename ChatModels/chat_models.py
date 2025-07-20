from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os


load_dotenv()


llm = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name (not model name)
    openai_api_version="2025-01-01-preview",
    temperature=0.7
)
chat_history = []
while True:
    user_input = input('You :' )
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit':
              break
    result = llm.invoke(chat_history)
    print("AI:", result.content)
    chat_history.append(AIMessage(content=result.content))

