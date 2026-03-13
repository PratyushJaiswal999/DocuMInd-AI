import fitz  # PyMuPDF
from typing import List, Dict, Any
import os

def extract_text_and_metadata_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text and basic metadata from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: The path to the PDF file.
        
    Returns:
        A list of dictionaries, where each dictionary represents a page
        and contains the 'text' and 'metadata' (like page number).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    documents = []
    try:
        # Open the document
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Basic metadata
            metadata = {
                "source": pdf_path,
                "page": page_num + 1,
            }
            
            # We only append pages that actually have text
            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": metadata
                })
                
        doc.close()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        
    return documents

def get_chunks(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    A simple chunking function. 
    In a real production app, you might use LangChain's RecursiveCharacterTextSplitter,
    but this provides a clean, dependency-light way to chunk the extracted text.
    """
    chunks = []
    
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        
        # Simple overlap chunking
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "metadata": metadata.copy() # Keep the metadata attached to each chunk
            })
            
            start += (chunk_size - chunk_overlap)
            
    return chunks

if __name__ == "__main__":
    # A simple test to ensure the script runs
    print("Extraction module loaded successfully.")
