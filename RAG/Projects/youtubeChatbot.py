from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI 
from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()  
video_id = "640KMYtxCeI"
ytt_api = YouTubeTranscriptApi()
transcript = ytt_api.fetch(video_id)

full_text = " ".join(snippet.text for snippet in transcript)

# print(full_text)




# split the text into chunks 

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200 )
chunks = splitter.create_documents([full_text])
print(len(chunks))
#initialize the embedding model

api_key = os.getenv("embedding-model-Api_Key")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("EMBEDDING_MODEL_VERSION")

embedding_model = AzureOpenAIEmbeddings(
    deployment=deployment,
    model="text-embedding-ada-002",  # Usually this
    openai_api_base=endpoint,
    openai_api_type="azure",
    openai_api_key=api_key,
    openai_api_version=api_version,
    chunk_size=1000
)
# generate the emebedding

#store the embedding into vector db
vector_store= FAISS.from_documents(chunks,embedding_model,)
vector_store.index_to_docstore_id
#initialize the gpt 4 llm

# retrieve the top 5 vector 

# pass it into model 

# genearte the response