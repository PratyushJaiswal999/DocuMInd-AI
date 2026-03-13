<div align="center">
  <h1>🧠 DocuMind AI</h1>
  <p><strong>A Production-Ready RAG Chatbot for Document Intelligence</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/Streamlit-1.30+-red.svg" alt="Streamlit">
    <img src="https://img.shields.io/badge/LLM-Groq_%7C_Llama_3.1-orange.svg" alt="Groq">
    <img src="https://img.shields.io/badge/Vector_DB-Chroma-green.svg" alt="ChromaDB">
  </p>
</div>

<br/>

DocuMind AI is a fast, robust, and visually polished Retrieval-Augmented Generation (RAG) application. It allows users to upload PDF documents and instantly chat with them, extracting facts, summaries, and answering specific questions based entirely on the document's context.

Website URL :- https://docu-mind-ai-by-pratyush.streamlit.app/

## ✨ Features

- **Blazing Fast Inference:** Powered by [Groq](https://groq.com/) and specifically optimized for the `llama-3.1-8b-instant` model.
- **Persistent Vector Storage:** Uses ChromaDB to locally store and index high-quality document embeddings (`BAAI/bge-base-en-v1.5`).
- **Smart Context Retrieval:** Intelligent chunking and retrieval ensure the LLM only answers based on factual data from your document, avoiding hallucinations. 
- **Premium UI/UX:** Built on Streamlit but heavily customized with raw CSS for a beautiful dark-mode interface, glassmorphism sidebar, distinct chat bubbles, and animated typing indicators.
- **Hash-Based Caching:** Automatically computes MD5 hashes of uploaded files to prevent redundant re-processing of the same document.

## 🏗️ Architecture

1. **Ingestion (`extraction.py`):** Uses PyMuPDF (`fitz`) to extract text and metadata from uploaded PDFs.
2. **Embedding (`embeddings.py`):** Converts text chunks into dense vectors using HuggingFace BGE embeddings offline (no API cost). 
3. **Storage (`vector_store.py`):** Embeddings are stored efficiently on disk in ChromaDB.
4. **Inference (`inference.py`):** A custom LangChain Expression Language (LCEL) chain retrieves the top-3 most relevant semantic chunks and streams the context to the Groq API for the final answer.
5. **Frontend (`app.py`):** The Streamlit orchestrator that ties the backend together into a seamless web experience.

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/documind-ai.git
cd documind-ai
```

### 2. Set up the environment
Create a virtual environment and install dependencies:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 4. Run the Application
```bash
streamlit run src/app.py
```
Open your browser to `http://localhost:8501`.

## 📂 Project Structure
```text
📦 documind-ai
 ┣ 📂 data           # Transitory storage for uploaded PDFs
 ┣ 📂 db_storage     # Persistent ChromaDB vector database
 ┣ 📂 src
 ┃ ┣ 📜 app.py           # Streamlit UI & Orchestrator
 ┃ ┣ 📜 embeddings.py    # HuggingFace BGE initialization
 ┃ ┣ 📜 extraction.py    # PyMuPDF text & chunk extraction
 ┃ ┣ 📜 inference.py     # Groq LLM & LCEL RAG Chain
 ┃ ┣ 📜 main.py          # Backend pipeline execution
 ┃ ┗ 📜 vector_store.py  # ChromaDB collection & query management
 ┣ 📜 .env           # Secrets (Gitignored)
 ┣ 📜 .gitignore     
 ┗ 📜 requirements.txt
```

## 🛠️ Built With
- **[Streamlit](https://streamlit.io/)** — Web Frontend
- **[LangChain](https://www.langchain.com/)** — Orchestration & Chains
- **[ChromaDB](https://www.trychroma.com/)** — Vector Database
- **[Groq](https://groq.com/)** — Ultra-fast LLM Inference
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — PDF Parsing
- **[SentenceTransformers](https://sbert.net/)** — Local Embeddings
