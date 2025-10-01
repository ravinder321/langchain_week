import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# ========== CONFIG ==========
load_dotenv()
CHUNKS_FILE = "data/chunks.json"
CHROMA_DIR = "./chroma_db"

# ========== LOAD CHUNKS ==========
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert json into LangChain Document objects
docs = [Document(page_content=item["text"], metadata=item["metadata"]) for item in data]
print(f"✅ Loaded {len(docs)} documents from {CHUNKS_FILE}")

# ========== CREATE EMBEDDINGS ==========
# Free HuggingFace model (downloads on first use, ~100MB)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ========== BUILD VECTORSTORE ==========
vectordb = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)

# Save to disk
vectordb.persist()
print(f"✅ Vector DB created and saved at {CHROMA_DIR}")

# ========== TEST RETRIEVAL ==========
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

query = "Who won the battle of Morlaix?"
results = retriever.get_relevant_documents(query)

print(f"\n🔍 Query: {query}")
for i, r in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(r.page_content[:300], "...")
    print("Metadata:", r.metadata)
