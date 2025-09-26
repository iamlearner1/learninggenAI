from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)
sample_text = "this is a sample line of text"
documents = [

    "this is my sample first line of text",
    "this is my sample second line of the text",
    "this is my sample third line of the text"
]

# result = embedding.embed_query(sample_text)
result = embedding.embed_documents(documents)

print(result)

