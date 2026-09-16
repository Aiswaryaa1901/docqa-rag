import chromadb
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="documents")

question = "What are the Functional Modules in this document?"
question_embedding = embedding_model.encode(question)

print("QUESTION:", question)
print("First 5 embedding numbers:", question_embedding[:5])

results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=2,
    where={"filename": "CN MP.pdf"}
)

print(results)