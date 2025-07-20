from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Get from .env or hardcode for testing
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Azure endpoint
api_version = os.getenv("OPENAI_API_VERSION")  # e.g., 2023-05-15
deployment = os.getenv("text-embedding-model")  # e.g., text-embedding-ada-002

# ✅ Use `AzureOpenAIEmbeddings` parameters EXACTLY like this:
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=api_base,             # ✅ This is the key that works!
    api_key=api_key,
    deployment=deployment,
    openai_api_version=api_version
)
print("Using deployment:", embeddings.deployment)
Documents = [
    "Hello, my name is Kushal Sharma, and I am learning about vector search.",
    "Vector search is a technique used to find similar items in a dataset based on their vector representations.",
    "It is commonly used in information retrieval, recommendation systems, and natural language processing tasks.",
]

query = "What is vector search?"

document_embedding = embeddings.embed_documents(Documents)
query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity(
    [query_embedding],
    document_embedding
)
most_similar_index = similarity_scores[0].argmax()
print("Similarity scores:", most_similar_index,similarity_scores[0])
# print("Most similar document:", Documents[most_similar_index])
