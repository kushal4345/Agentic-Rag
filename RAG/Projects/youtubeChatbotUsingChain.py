from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI 
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
import os

# --- Step 1: Load Environment and Prepare Data ---
load_dotenv()  
video_id = "GDrBIKOR01c"
ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.fetch(video_id)
full_text = " ".join(snippet.text for snippet in transcript)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200 )
chunks = splitter.create_documents([full_text])
print(f"Transcript split into {len(chunks)} chunks.")

# --- Step 2: Initialize Models and Vector Store ---
embedding_model = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),
        openai_api_version=os.getenv("EMBEDDING_MODEL_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

vector_store = FAISS.from_documents(chunks,embedding_model)


# The retriever component to fetch relevant documents
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# The language model component
model = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_version="2024-02-01", 
    temperature=0.7,
    max_tokens=500
)

# The prompt template component
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      <context>
      {context}
      </context>
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

# A function to format the retrieved documents into a single string
def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

#parallel chain which combine the retiever and question
parallel_chain = (
{
    "context": retriever |RunnableLambda(format_docs),  # the q will pass to both the docs retriever and question 
    "question": RunnablePassthrough()
}
)
#combine the parrallel chain with sequential chain
main_chain = parallel_chain|prompt|model|StrOutputParser()

#Invoke the Chain and Print the Result ---
question = "gimme the summary of this video"
print("\nInvoking RAG chain...")
final_answer = main_chain.invoke(question)

print("\n--- Answer ---")
print(final_answer)