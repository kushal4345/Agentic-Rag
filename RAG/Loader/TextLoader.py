from langchain_community.document_loaders import TextLoader

loader = TextLoader("RAG/Loader/document.txt", encoding='utf-8')
documents = loader.load()

print(type(documents))
print(len(documents))
print(documents[0].page_content)
