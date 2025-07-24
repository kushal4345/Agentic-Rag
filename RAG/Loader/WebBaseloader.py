from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) LangChainBot/1.0"
loader = WebBaseLoader('https://en.wikipedia.org/wiki/2020%E2%80%932021_China%E2%80%93India_skirmishes')
documents = loader.load()
model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)
prompt = PromptTemplate(
    template = "Answer the following question {question} from the following text {text}",
    input_variables = ["question","text"]
)
parser = StrOutputParser()

chain = prompt|model | parser

result = chain.invoke({'question':'topic of the given text','text' : documents[0].page_content})
print(result)
# print(documents[0])
