import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("EduRAG - Study Assistant")

f = st.file_uploader("Upload PDF", type="pdf")
if f and st.button("Process PDF"):
    r = requests.post(f"{API}/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
    st.success(f"Indexed {r.json()['chunks']} chunks")

q = st.text_input("Ask a question")
if q:
    r = requests.post(f"{API}/ask", json={"question": q}).json()
    st.write(r["answer"])
    st.caption("Sources: " + ", ".join(r["sources"]))