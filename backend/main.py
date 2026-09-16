from fastapi import FastAPI, File, UploadFile
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import io

app = FastAPI()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/")
def read_root():
    return {"message": "doqua-rag backend is running!"}

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk =  " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    pdf = PdfReader(io.BytesIO(contents))

    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    chunks = chunk_text(text)
    embeddings = embedding_model.encode(chunks)

    return {
    "filename": file.filename,
    "char_count": len(text),
    "num_chunks": len(chunks),
    "embeddings_shape": embeddings.shape,
    "first_chunk_preview": chunks[0][:100],
    "first_embedding_preview": embeddings[0][:5].tolist()
} 
  