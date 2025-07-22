from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Optional

prompt1 = PromptTemplate(
    template = "you ar a helpful {domain} assistant that will give the detail summary of {topic} in a good format and with a good structure",
    input_variables = ["domain", "topic"]
)

class Summary(BaseModel):
    """A class to represent a summary."""
    domain: str = Field(..., description="Domain of the summary")
    summary: str = Field(..., description="Topic of the summary")
    birth: str = Field(..., description="Detailed summary of the topic")
    key_points: Optional[str] = Field(None, description="Key points extracted from the summary")
    controversial: Optional[str] = Field(None, description="Controversial aspects of the topic")



prompt2 = PromptTemplate(
    template = "you have to extract key points from the following text: {text}",
    input_variables = ["text"]
)
# Load the Azure Chat OpenAI model
load_dotenv()

#call the model

model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens= 500
)

# Define the input variables
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"domain": "story teller", "topic": "lord rama"})
print(result)
print(chain.get_graph().print_ascii())  # This will print the graph of the chain

