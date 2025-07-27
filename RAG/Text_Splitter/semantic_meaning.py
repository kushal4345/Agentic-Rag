from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv

text_splitter = SemanticChunker(
OpenAIEmbeddings() , breakpoint_threshold_type="standard deviation" , #deviation means comaparing each chunk and check similarity
breakpoint_threshold_amount=1
)

sample = """
write your text here ,add any text 
"""

