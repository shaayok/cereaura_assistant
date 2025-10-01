import os, hashlib
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

DATA_PATH = Path("data")
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "autism_bot")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# load docs
docs = []
if DATA_PATH.exists():
    docs.extend(PyPDFDirectoryLoader(str(DATA_PATH)).load())
    for docx_file in DATA_PATH.glob("*.docx"):
        docs.extend(Docx2txtLoader(str(docx_file)).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

added = 0
for chunk in chunks:
    text = chunk.page_content.strip()
    if not text:
        continue
    uid = hashlib.md5(text.encode("utf-8")).hexdigest()
    embedding = client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding
    collection.upsert(
        documents=[text],
        embeddings=[embedding],
        metadatas=[chunk.metadata],
        ids=[uid]
    )
    added += 1

print(f"✅ Added {added} chunks into {COLLECTION_NAME}")
