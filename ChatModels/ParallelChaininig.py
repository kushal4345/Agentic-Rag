from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from langchain.schema.runnable import RunnableParallel
load_dotenv()
import os

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",  # model name must start with "models/"
    google_api_key=os.getenv("GOOGLE_API_KEY")  # or hardcode your key
)
llm2 = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name (not model name)
    openai_api_version="2025-01-01-preview",
    temperature=0.7
)
prompt1 = PromptTemplate(
    template="generate short notes from the following text \n {text}",
    input_variables=["text"]
)
prompt2 = PromptTemplate(
    template="generate 5 short question answer rom the folowing text \n {text}",
    input_variables=["text"]
)
prompt3 =  PromptTemplate(
    template="merge the provided notes and quiz into single document notes->\n {notes} quiz -> {quiz} ",
    input_variables=["notes", "quiz"]
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | llm2 | parser,
    'quiz': prompt2 | llm | parser
})
merge_chain = prompt3 | llm2 | parser
chain = parallel_chain | merge_chain
result = chain.invoke({"text": "The quick brown fox jumps over the lazy dog. This is a classic example of a pangram, which is a sentence that contains every letter of the alphabet at least once."})
print(result)
print(chain.get_graph().print_ascii())  # This will print the graph of the chain
# The output will contain the merged notes and quiz based on the input text.