from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

NOTES_DIR = Path("notes")
DB_DIR = "chroma_db"

docs = []
for pdf in NOTES_DIR.glob("*.pdf"):
    docs.extend(PyPDFLoader(str(pdf)).load())
print(f"loaded {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"made {len(chunks)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
print("done, saved to chroma_db")