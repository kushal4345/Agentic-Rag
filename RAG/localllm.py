from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


loader = PyPDFLoader("RAG/AshishResume.pdf")  # <- 
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
documents = splitter.split_documents(docs)


embeddings = OllamaEmbeddings(model="nomic-embed-text")


vectorstore = FAISS.from_documents(documents, embeddings)


vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)



llm = Ollama(model="llama3.2:3b")

summary_prompt = "Summarize the following document in a clear and concise way:\n\n"
full_text = " ".join([doc.page_content for doc in documents])

summary = llm.invoke(summary_prompt + full_text)
print("\n📄 PDF Summary:\n", summary, "\n")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

print("RAG Chatbot ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    response = qa_chain.invoke({"question": query})
    print("Bot:", response["answer"])
