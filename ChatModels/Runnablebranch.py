from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnableBranch,RunnablePassthrough
load_dotenv()
prompt1 = PromptTemplate(
    template = "generate the document on the given topic  {topic}",
    input_variables=["topic"]
)
model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()
chain = RunnableSequence(prompt1, model, parser)
prompt2 = PromptTemplate(
    template = " summarize the given text {text}",
    input_variables=["text"])

branch = RunnableBranch(
    (lambda x : len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(chain, branch)
result = chain.invoke({"topic": "Generative AI"})
print(result)