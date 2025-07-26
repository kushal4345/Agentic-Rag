from langchain.text_splitter import RecursiveCharacterTextSplitter

# text = """
# One important consideration is that without proper synchronization, the Producer and Consumer may interfere with each other. For example, two producers writing data simultaneously could overwrite each other's data. Similarly, a consumer trying to read from an empty buffer could cause an error or crash. Thus, synchronization ensures data consistency, prevents race conditions, and avoids deadlocks or starvation.

# """
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 100.,
#     chunk_overlap = 0
# )
# result =splitter.split_text(text)
# print(result[0])

# this text splitter can be useful for other languages unlike hindi or any other human language it would work with pyton and other lang

text = """
num = int(input("Enter a number: "))

if num <= 1:
    print("Not a prime number")
else:
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            print("Not a prime number")
            break
    else:
        print("It is a prime number")

"""

splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size = 150,
    chunk_overlap = 0,
    language = "python"
)

chunk = splitter.split_text(text)
print(chunk)
print(chunk[0])