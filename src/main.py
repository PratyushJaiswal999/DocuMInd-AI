import os
from extraction import extract_text_and_metadata_from_pdf, get_chunks
from embeddings import create_embeddings_for_texts
from vector_store import insert_documents, query_vector_store
from inference import generate_answer

def run_ingestion_pipeline(pdf_path: str):
    """
    Step 1: Ingests a new PDF into ChromaDB.
    """
    print(f"\n--- Starting Ingestion Pipeline for {pdf_path} ---")
    
    # Extract
    extracted_docs = extract_text_and_metadata_from_pdf(pdf_path)
    if not extracted_docs:
        raise RuntimeError(
            "No text could be extracted from this PDF. "
            "It may be a scanned/image-based PDF. "
            "Only text-based PDFs are currently supported. Please try a different document."
        )
        
    print(f"Extracted {len(extracted_docs)} pages.")
    
    # Chunk
    chunks = get_chunks(extracted_docs, chunk_size=3000, chunk_overlap=300)
    print(f"Created {len(chunks)} chunks.")
    
    # Embed
    texts_to_embed = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings_for_texts(texts_to_embed)
    print(f"Generated {len(embeddings)} embeddings.")
    
    # Store
    insert_documents(chunks, embeddings)
    print("Ingestion Complete. Ready for Retrieval.\n")

def run_retrieval_and_generation(query: str):
    """
    Step 2: Answers a user query using the RAG pipeline.
    """
    print(f"\n--- Starting RAG Pipeline ---")
    print(f"Question: '{query}'")
    
    # 1. Embed the user's query
    query_vector = create_embeddings_for_texts([query])[0]
    
    # 2. Retrieve relevant chunks
    results = query_vector_store(query_vector, n_results=3)
    
    # Reformat ChromaDB results into a simpler list of dicts for our generator
    retrieved_chunks = []
    if results and results.get("documents"):
        # Chroma returns lists of lists for n_results
        docs = results["documents"][0] 
        metas = results["metadatas"][0]
        for i in range(len(docs)):
            retrieved_chunks.append({
                "text": docs[i],
                "metadata": metas[i]
            })
            
    if not retrieved_chunks:
        print("No relevant context found in the database. Please ingest some documents first.")
        return
        
    print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
    
    # 3. Generate answer
    answer = generate_answer(query, retrieved_chunks)
    print(f"\nFinal Answer:\n{answer}")

if __name__ == "__main__":
    
    print("=========================================")
    print("     Clean RAG Production Pipeline       ")
    print("=========================================\n")
    
    # Example usage flow:
    # 
    # 1. Provide a PDF to ingest
    # Resolve the path relative to the location of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file = os.path.join(script_dir, "..", "data", "multimodal_sample.pdf")
    
    if os.path.exists(pdf_file):
        run_ingestion_pipeline(pdf_file)
    else:
        print(f"Could not find PDF file at: {os.path.abspath(pdf_file)}")
        print("Please place 'multimodal_sample.pdf' in the data folder!")
        
    # 
    # 2. Ask a question
    run_retrieval_and_generation("What is the main topic of the document?")
    # print("Pipeline ready. Uncomment the ingestion and retrieval steps in main.py to run.")
