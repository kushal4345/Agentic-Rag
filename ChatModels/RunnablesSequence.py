from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

#load environment variables
load_dotenv()

#prompt 

prompt = PromptTemplate(
    template= "you are a helpful {domain} assistant that will provide a detailed summary of {topic} in a well-structured format.",
    input_variables=["domain", "topic"]
)
# Load the Azure Chat OpenAI model
model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()
# Create a RunnableSequence to process the input through the prompt, model, and parser
chain = RunnableSequence(prompt, model, parser)

# Invoke the chain with input variables
result = chain.invoke({"domain": "story teller", "topic": "lord Krishna"})
print(result)
print(chain.get_graph().print_ascii())  # This will print the graph of the chain
    

