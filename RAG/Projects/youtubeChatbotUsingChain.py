from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import os
import pinecone
import json
import pinecone  # make sure this is installed: pip install pinecone-client[grpc]

# --- Step 1: Load environment variables ---
load_dotenv()

# --- Step 2: Define output schema using Pydantic ---
class VideoSummary(BaseModel):
    summary: str = Field(..., description="Short overview of the video content.")
    motive: str = Field(..., description="Purpose or motive of the video.")
    key_points: List[str] = Field(..., description="List of key points covered in the video.")

parser = PydanticOutputParser(pydantic_object=VideoSummary)

# --- Step 3: Fetch transcript ---
video_id = "GDrBIKOR01c"
ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.fetch(video_id)
full_text = " ".join(snippet.text for snippet in transcript)

# --- Step 4: Split transcript into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([full_text])
print(f"Transcript split into {len(chunks)} chunks.")

# --- Step 5: Initialize embedding model ---
embedding_model = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),
    openai_api_version=os.getenv("EMBEDDING_MODEL_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# --- Step 6: Initialize Pinecone and create index ---
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")  # or your environment

# Create or connect to index
index_name = "nerv"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536)

index = pinecone.Index(index_name)

# Use it in Langchain
vector_store = LangchainPinecone.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=index_name,
  
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- Step 8: Initialize LLM model ---
model = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2024-02-01",
    temperature=0.7,
    max_tokens=800
)

# --- Step 9: Prompt Template ---
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Use ONLY the following context to answer.
If context is not enough, say "insufficient context".

<context>
{context}
</context>

{format_instructions}

Question: {question}
""",
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# --- Step 10: Format retrieved documents ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Step 11: Assemble the RAG pipeline ---
parallel_chain = {
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
}

main_chain = parallel_chain | prompt | model | parser

# --- Step 12: Ask the question ---
question = "Summarize this video with summary, motive, and key points."
print("\nInvoking RAG chain...")

response = main_chain.invoke(question)

# --- Step 13: Display response in pretty JSON ---

print(json.dumps(response.dict(), indent=2))
