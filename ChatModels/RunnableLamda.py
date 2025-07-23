from langchain_core.runnables import Runnable,RunnableLambda,RunnablePassthrough,RunnableSequence,RunnableParallel
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

#function which will pass to the RunnableLambda
def joke_length(text):
    return len(text.split())

#prompt to generate a joke
prompt = PromptTemplate(
    template = "generate a joke on a given topic {topic}",  
    input_variables=["topic"]
)
# Load the Azure Chat OpenAI model
model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()
# Create a RunnableLambda to calculate the length of the joke
joke_gen_chain = RunnableSequence(prompt, model, parser)
joke_count_length = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length': RunnableLambda(joke_length)
})
# Combine the joke generation and length calculation into a single chain

result = RunnableSequence(joke_gen_chain, joke_count_length)

# Invoke the chain with input variables
result = result.invoke({"topic": "Engineering"})

print(result)