from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough,RunnableParallel
load_dotenv()
prompt1 = PromptTemplate(
    template = " generate a joke on a given topic {topic}",
    input_variables=["topic"])
model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)
parser = StrOutputParser()
prompt2 = PromptTemplate(
    template =  "explain the following joke in simple terms: {text}",
    input_variables=["text"])

joke_gen_chain = RunnableSequence(prompt1, model, parser)
explain_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})
result = joke_gen_chain | explain_chain
result = result.invoke({"topic": "Engineering"})
print(result)