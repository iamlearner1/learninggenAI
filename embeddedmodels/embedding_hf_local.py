from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of india"

documents = [
    "Mumbai is a major city in India",
    "New Delhi serves as the capital city of India",
    "India has several metropolitan cities including Delhi",
    "Delhi is known for its historical landmarks",
    "The capital of India is New Delhi",
    "Chennai is a coastal city in South India",
    "Delhi has a population of over 18 million people",
    "India's political center is located in Delhi",
    "Tourists often visit Delhi for its culture and history",
    "Kolkata was once the capital of British India"
]


vector_query = embedding.embed_query(text)
vector_documents = embedding.embed_documents(documents)
# print("Vector query of the text : ",vector_query)
# print("Vector of all the documents : ",vector_documents)
result_similarity = cosine_similarity([vector_query],vector_documents)[0]
# print("Result similarity : ",result_similarity)

index, score = sorted(list(enumerate(result_similarity)),key = lambda x : x[1])[-1]
print("Matching document with the given input text is : ",documents[index],"with the similarity of : ",score)
