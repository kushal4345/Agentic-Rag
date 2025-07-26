from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('RAG/Text_Splitter/newAdmit.pdf')

docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

text = splitter.split_documents(docs)
print(text[0].page_content)