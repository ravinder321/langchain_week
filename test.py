from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vec = emb.embed_query("Hello world")
print(len(vec))  # should print embedding size, e.g., 384
