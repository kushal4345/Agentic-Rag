from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI 
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()  
video_id = "GDrBIKOR01c"
ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.fetch(video_id)

full_text = " ".join(snippet.text for snippet in transcript)

# print(full_text)


# split the text into chunks 

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200 )
chunks = splitter.create_documents([full_text])
print(len(chunks))

#inirtialize the model
embedding_model = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_key=os.getenv("EMBEDDING_MODEL_API_KEY"),
        openai_api_version=os.getenv("EMBEDDING_MODEL_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
 

vector_store= FAISS.from_documents(chunks,embedding_model,)
vector_store.index_to_docstore_id


# # retrieve the top 5 vector 
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
results = retriever.invoke("What are AI agents?")

#initialize the llm
model = AzureChatOpenAI(
    deployment_name="gpt-4o",  # Your Azure deployment name
    openai_api_version="2025-01-01-preview",
    temperature=0.7,
    max_tokens=500
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question= "gimme the summary of this video"
retrieved_docs= retriever.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = model.invoke(final_prompt)
print(answer.content)
# # genearte the response