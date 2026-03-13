import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any

# Where the ChromaDB database will be stored locally
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'db_storage')
COLLECTION_NAME = "my_rag_collection"

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Initializes and returns a ChromaDB client that persists data to disk.
    """
    print(f"Connecting to ChromaDB at: {PERSIST_DIRECTORY}")
    
    # Ensure the directory exists
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # The 'anonymized_telemetry' parameter turns off telemetry
    client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY, 
        settings=Settings(anonymized_telemetry=False)
    )
    return client

def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = COLLECTION_NAME) -> chromadb.Collection:
    """
    Retrieves or creates a ChromaDB collection.
    By default, Chroma uses "all-MiniLM-L6-v2" if no embedding function is provided.
    However, we will handle the embeddings ourselves (via BGE in Python) and pass
    the floating point vectors directly to Chroma as a list of lists.
    """
    # Create or get collection
    # Note: We are not explicitly providing an embedding_function here.
    # We will compute the BGE embeddings offline and pass them to `.add()` directly.
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def insert_documents(chunks: list[dict[str, any]], embeddings: list[list[float]]):
    """
    Inserts a list of document chunks and their corresponding embeddings into ChromaDB.
    Clears the existing collection first to ensure we only query the newly uploaded document!
    """
    client = get_chroma_client()
    
    # 1. Delete the old collection to wipe existing context
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted old document context from vectors.")
    except Exception:
        pass # Collection might not exist yet
        
    # 2. Re-create the collection fresh
    collection = get_or_create_collection(client)
    
    # Chroma requires distinct IDs for every document snippet
    ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
    
    # Extract just the raw strings for the documents, and mapping the metadata 
    # to be searchable later.
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    print(f"Adding {len(chunks)} chunks to Chroma DB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,   # The pre-computed BGE raw vectors
        documents=documents,     # The raw text
        metadatas=metadatas      # E.g. {"source": "my_pdf.pdf", "page": 1}
    )
    print("Database insert complete.")

def query_vector_store(query_vector: list[float], n_results: int = 5) -> dict:
    """
    Queries ChromaDB using a pre-computed vector.
    Automatically caps n_results to the actual document count to prevent errors.
    """
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    # CRITICAL: ChromaDB throws an error if n_results > total docs in collection
    actual_count = collection.count()
    if actual_count == 0:
        return {"documents": [], "metadatas": []}
    
    safe_n_results = min(n_results, actual_count)
    
    # The `query` method supports passing the query_embeddings directly
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=safe_n_results
    )
    
    return results

if __name__ == "__main__":
    client = get_chroma_client()
    print("ChromaDB Client successfully initialized.")
