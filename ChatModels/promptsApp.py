from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = "gpt-4o" # e.g., "gpt-35-turbo"
api_version = os.getenv("OPENAI_API_VERSION")     # e.g., "2023-05-15"

# Initialize the Azure OpenAI chat model
model = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    deployment=deployment,
    openai_api_version=api_version
)
st.header("Chat with Azure OpenAI")
user_input = st.text_input("Enter your message:")

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)

