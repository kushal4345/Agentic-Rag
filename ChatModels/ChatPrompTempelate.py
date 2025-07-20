from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7
)

# Define chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} assistant."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Load chat history
chat_history = []
with open('chathistory.txt', 'r') as file:
    for line in file:
        if line.startswith('HumanMessage'):
            chat_history.append(HumanMessage(content=line.split(':', 1)[1].strip()))
        elif line.startswith('AiMessage'):
            chat_history.append(AIMessage(content=line.split(':', 1)[1].strip()))

# Generate prompt value from template
prompt_value = prompt.invoke({'chat_history': chat_history, 'input': 'i want my full refund', 'domain': 'e-commerce'})

# Pass the prompt to the LLM
response = llm.invoke(prompt_value)

# Print the LLM's response
print("AI:", response.content)
