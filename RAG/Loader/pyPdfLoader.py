from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
loader = PyPDFLoader('RAG/Loader/newAdmit.pdf')
documents = loader.load()
print(documents[0])

