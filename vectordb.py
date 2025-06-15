import os
import faiss
from langchain_community.vectorstores import FAISS
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

class TransformersEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        local_model = SentenceTransformer('local-all-MiniLM-L6-v2')
        if not local_model:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.tokenizer = local_model.tokenizer
            self.model = local_model._first_module()
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            return embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embedding = TransformersEmbeddings()
index_embedding = TransformersEmbeddings()

db_path = "data/vector_db"
db_index_path = "data/db_index.faiss"

def init_vector_strore(url:str, summary: str):
    db_path = f"{db_path}/{url.replace('http://', '').replace('https://', '').replace('/', '_')}.faiss"
    # If the database exists, load it; otherwise, create a new one
    if not FAISS.exists_local(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Save to db_index
    index_entry = [f"URL: {url}", f"DB Path: {db_path}", f"Summary: {summary}"]
    if not FAISS.exists_local(db_index_path):
        os.makedirs(os.path.dirname(db_index_path), exist_ok=True)
    db_index = FAISS.load_local(db_index_path, index_embedding) if FAISS.exists_local(db_index_path) else FAISS.from_texts([], index_embedding)
    db_index.add_texts(index_entry)
    db_index.save_local(db_index_path)
    db_index.save_local(db_path)

def save_to_vector_store(url:str, chunks: list[str]):
    if not chunks:
        return  # No chunks to save
    # use url as the db fiel name
    db_path = f"{db_path}/{url.replace('http://', '').replace('https://', '').replace('/', '_')}.faiss"
    # If the database exists, load it; otherwise, create a new one
    
    if not FAISS.exists_local(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db = FAISS.load_local(db_path, embedding) if FAISS.exists_local(db_path) else FAISS.from_texts([], embedding)
    db.add_texts(chunks)
    db.save_local(db_path)

def query_knowledge(prompt: str) -> str:
    # Look for related db_path in db_index first
    db_index = FAISS.load_local(db_index_path, index_embedding)
    index_docs = db_index.similarity_search(prompt, k=5)
    related_db_paths = [doc.page_content.split("DB Path: ")[1].split(",")[0] for doc in index_docs]
    
    docs = []
    for path in related_db_paths:
        db = FAISS.load_local(path, embedding)
        docs.extend(db.similarity_search(prompt, k=5))
    context = "\n".join([doc.page_content for doc in docs])

    return context

def del_db(url: str):
    """Delete the vector database and db_index entry for the given URL."""
    db_path = f"data/vector_db/{url.replace('http://', '').replace('https://', '').replace('/', '_')}.faiss"
    # Delete the vector database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Remove the entry from db_index
    if FAISS.exists_local(db_index_path):
        db_index = FAISS.load_local(db_index_path, index_embedding)
        index_docs = db_index.similarity_search(url, k=1)
        for doc in index_docs:
            if url in doc.page_content:
                db_index.delete_texts([doc.page_content])
                break
        db_index.save_local(db_index_path)

    return True
