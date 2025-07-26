from langchain.text_splitter import CharacterTextSplitter


text = """
The Producer-Consumer Problem is one of the most common examples used to demonstrate synchronization in operating systems and concurrent programming. It models a situation where two types of processes, the Producer and the Consumer, interact with a shared, finite-size resource called a buffer. The Producer generates data and places it into the buffer, while the Consumer removes data from the buffer for further processing. The key challenge in this problem is ensuring that the producer does not add data into a full buffer and the consumer does not remove data from an empty one, all while preventing data corruption through proper synchronization.

"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

text = splitter.split_text(text)
print(text)