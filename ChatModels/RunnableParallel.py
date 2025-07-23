from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
load_dotenv()
prompt1 = PromptTemplate(
    template = "generate a tweet on a given topic {topic}",
    input_variables=["topic"]
)
model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)

parser = StrOutputParser()
prompt2 = PromptTemplate(
    template = " generate the short and professional linkdin post on a given topic {topic}",
    input_variables=["topic"])

Parallelchain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkdin': RunnableSequence(prompt2, model, parser)
})

result = Parallelchain.invoke({"topic": "Generative AI"})
print(result)
# print(result['tweet'])
# print(result['linkdin'])
print(Parallelchain.get_graph().print_ascii())  # This will print the graph of the