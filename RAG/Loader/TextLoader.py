from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
loader = TextLoader("RAG/Loader/document.txt", encoding='utf-8')
documents = loader.load()

model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)
prompt = PromptTemplate(
    template = "summarize the given poen in 50 words{poem}",
    input_variables = ["poem"]
)
parser = StrOutputParser()

chain = prompt|model | parser

result = chain.invoke({'poem':documents[0].page_content})
print(result)
# print(len(documents))
# print(documents[0].page_content)
