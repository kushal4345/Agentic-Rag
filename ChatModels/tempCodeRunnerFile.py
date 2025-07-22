from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Optional
load_dotenv()

class Roadmap(BaseModel):
    domain: str
    topic: str
    roadmap: str
    Time:Optional[int] = Field(None, description="Time required to complete the roadmap in months")


prompt = PromptTemplate(
    template= "you are a helpfull {domain} assistant that will tell the roadmap of {topic} in a good format and with a good structure",
    input_variables = ["domain","topic"]
)

model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens= 500
)
structured_output = model.with_structured_output(Roadmap) # give structure to the output of the model
parser = StrOutputParser()
chain = prompt | model | parser | structured_output # This creates a chain that processes the input through the prompt, model, parser, and structured output.

result = chain.invoke({"domain": "teacher", "topic": "generative AI"})
print(result)
chain.get_graph().print_ascii()  # This will print the graph of the chain