import os   #system operation handle
import faiss   #to search similar embedding
import numpy as np   #handle numerical calculation
from pypdf import PdfReader   #extract text
from fastembed import TextEmbedding    #text to embeddings
from groq import Groq      #access ai model using api
from dotenv import load_dotenv   #load apikey

load_dotenv()
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
index = faiss.IndexFlatIP(384)
chunks, sources = [], []

def embed(texts):
    v = np.array(list(embedder.embed(texts)), dtype="float32")
    faiss.normalize_L2(v)
    return v

def add_pdf(path, name):
    text = ""
    for page in PdfReader(path).pages:
        text += (page.extract_text() or "") + "\n"
    new = [text[i:i + 800] for i in range(0, len(text), 700) if text[i:i + 800].strip()]
    if not new:
        return 0
    index.add(embed(new))
    chunks.extend(new)
    sources.extend([name] * len(new))
    return len(new)

def ask(question, k=4):
    if not chunks:
        return "Upload a PDF first.", []
    _, ids = index.search(embed([question]), k)
    ids = [i for i in ids[0] if i != -1]
    context = "\n\n".join(chunks[i] for i in ids)
    r = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Answer only from the context. If not found, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return r.choices[0].message.content, sorted({sources[i] for i in ids})