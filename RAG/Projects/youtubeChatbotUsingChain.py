from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os
import json  # ✅ Added for JSON parsing

# --- Step 1: Load Environment and Prepare Data ---
load_dotenv()

class YouTubeSummary(BaseModel):
    summary: str = Field(description="A brief summary of the video")
    motive: str = Field(description="The main motive of the video")
    points: List[str] = Field(description="Key points from the video")

video_id = "GDrBIKOR01c"
ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.fetch(video_id)

full_text = " ".join(snippet.text for snippet in transcript)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([full_text])
print(f"Transcript split into {len(chunks)} chunks.")

# --- Step 2: Initialize Models and Vector Store ---
embedding_model = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    openai_api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),
    openai_api_version=os.getenv("EMBEDDING_MODEL_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

model = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2024-02-01",
    temperature=0.7,
    max_tokens=700
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      Provide the output in this JSON format:
      {{
        "summary": "...",
        "motive": "...",
        "points": ["...", "...", "..."]
      }}

      <context>
      {context}
      </context>
      Question: {question}
    """,
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = {
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
}

main_chain = parallel_chain | prompt | model | StrOutputParser()

# --- Step 3: Invoke the Chain and Parse the Output ---
question = "Summarize this video with summary, motive, and key points in JSON."
print("\nInvoking RAG chain...")

final_answer = main_chain.invoke(question)

try:
    answer_dict = json.loads(final_answer)  # ✅ Use JSON parsing
    yt_summary = YouTubeSummary(**answer_dict)
    print("\n--- Final Structured Output ---")
    print(yt_summary.json(indent=2))
except Exception as e:
    print("\n❌ Failed to parse model output into YouTubeSummary:")
    print("🔹 Raw Output:\n", final_answer)
    print("🔸 Error:", str(e))
