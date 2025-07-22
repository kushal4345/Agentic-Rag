from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template= "you are a helpfull {domain} assistant that will tell the roadmap of {topic} in a good format and with a good structure",
    input_variables = ["domain","topic"]
)