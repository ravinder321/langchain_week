from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

os.makedirs("data", exist_ok=True)

# Load documents
loader = TextLoader("testing_data/sample.txt", encoding="utf-8")  # or TextLoader("sample.txt")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
#print(chunks)

#Save chunks
data = [ {"text": chunk.page_content, "metadata": chunk.metadata} for chunk in chunks ]
with open("data/chunks.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=3)

print(f"Loaded {len(docs)} docs and saved {len(chunks)} chunks.")
