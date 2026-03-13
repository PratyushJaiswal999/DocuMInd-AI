from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from typing import List

# Using a standard, performant BGE model for embeddings
# The "base" model is a good balance of speed and quality for local inference
# You could easily swap this for "BAAI/bge-large-en-v1.5" later if needed
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {'device': 'cpu'} # Change to 'cuda' or 'mps' if a GPU is available
encode_kwargs = {'normalize_embeddings': True} # True for cosine similarity

def get_bge_embeddings() -> HuggingFaceBgeEmbeddings:
    """
    Initializes and returns the HuggingFace BGE embeddings model via LangChain.
    """
    print(f"Loading embedding model: {model_name}...")
    embeddings_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("Embedding model loaded.")
    return embeddings_model

def create_embeddings_for_texts(texts: List[str]) -> List[List[float]]:
    """
    Given a list of strings, generates their vector embeddings using BGE.
    """
    embeddings_model = get_bge_embeddings()
    # Generates a vector for each string
    vectors = embeddings_model.embed_documents(texts)
    return vectors

if __name__ == "__main__":
    # A simple test to ensure the script runs and the model loads
    # Note: the first time this runs, it will download the model (~400MB)
    print("Initialize Embeddings...")
    model = get_bge_embeddings()
    test_vector = model.embed_query("This is a test document.")
    print(f"Test vector dimension: {len(test_vector)}")
