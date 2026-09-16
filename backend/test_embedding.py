from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "The quick brown fox jumps over the lazy dog"
embedding = model.encode(sentence)

print("Embedding shape:", embedding.shape)
print("First 10 numbers:", embedding[:10])