import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import rag

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
        t.write(await file.read())
    return {"chunks": rag.add_pdf(t.name, file.filename)}

@app.post("/ask")
def ask(body: Question):
    answer, srcs = rag.ask(body.question)
    return {"answer": answer, "sources": srcs}