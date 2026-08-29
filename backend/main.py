from fastapi import FastAPI, File, UploadFile
from pypdf import PdfReader
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "doqua-rag backend is running!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    pdf = PdfReader(io.BytesIO(contents))

    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return {"filename": file.filename, "char_count": len(text), "preview": text[:300]}