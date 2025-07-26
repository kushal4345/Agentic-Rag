from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
One important consideration is that without proper synchronization, the Producer and Consumer may interfere with each other. For example, two producers writing data simultaneously could overwrite each other's data. Similarly, a consumer trying to read from an empty buffer could cause an error or crash. Thus, synchronization ensures data consistency, prevents race conditions, and avoids deadlocks or starvation.

"""
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100.,
    chunk_overlap = 0
)
result =splitter.split_text(text)
print(result[0])