from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
loader = DirectoryLoader(
    directory_path="/home/user/Documents/"  #load all the pdf present inside the directory 
    glob='*.pdf',
    loader_cls= PyPDFLoader
)
documents = loader.load()
print(documents[0])
