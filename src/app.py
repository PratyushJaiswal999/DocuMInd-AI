import streamlit as st
import os
import hashlib
import datetime
from dotenv import load_dotenv
from main import run_ingestion_pipeline
from inference import generate_answer
from embeddings import create_embeddings_for_texts
from vector_store import query_vector_store

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="centered")


def inject_css():
    st.html("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

      /* ── Background ── */
      .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

      /* ── Header ── */
      .app-header { text-align: center; padding: 2rem 0 0.5rem; }
      .app-header h1 {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;
      }
      .app-header p { color: #94a3b8; font-size: 1rem; margin: 0; }

      /* ── Document pill ── */
      .doc-pill {
        display: inline-block; background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.4); color: #a78bfa;
        border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.85rem; margin-bottom: 0.5rem;
      }

      /* ── Sidebar ── */
      section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.9) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.07);
      }
      section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3 { color: #a78bfa !important; }

      /* ── POINT 2: Distinct user vs AI chat bubbles with right/left alignment ── */
      /* Hide avatar icons */
      [data-testid="stChatMessageAvatarUser"],
      [data-testid="stChatMessageAvatarAssistant"],
      [data-testid="stChatMessage"] > div:first-child { display: none !important; }

      /* Make message containers flex so we can align left/right */
      [data-testid="stChatMessage"] {
        display: flex !important;
        flex-direction: column !important;
        border-radius: 14px !important;
        margin-bottom: 0.6rem;
        max-width: 80%;
        padding: 0.8rem 1rem !important;
      }

      /* User messages → right side */
      [data-testid="stChatMessage"][data-role="user"], 
      [data-testid="stChatMessage"]:has(img[alt="user"]) {
        background: linear-gradient(135deg, rgba(124,58,237,0.45), rgba(79,70,229,0.35)) !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
        border-radius: 18px 18px 4px 18px !important;
        align-self: flex-end;
        margin-left: auto;
        margin-right: 0;
      }

      /* AI messages → left side */
      [data-testid="stChatMessage"][data-role="assistant"],
      [data-testid="stChatMessage"]:has(img[alt="assistant"]) {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(96,165,250,0.25) !important;
        border-radius: 18px 18px 18px 4px !important;
        align-self: flex-start;
        margin-right: auto;
        margin-left: 0;
      }

      [data-testid="stChatMessage"] p { color: #e2e8f0 !important; font-size: 0.97rem; line-height: 1.6; }

      /* ── POINT 6: Fixed-height scrollable chat area ── */
      [data-testid="stVerticalBlock"] > div:has([data-testid="stChatMessage"]) {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 4px;
        scroll-behavior: smooth;
      }
      [data-testid="stVerticalBlock"] > div:has([data-testid="stChatMessage"])::-webkit-scrollbar {
        width: 4px;
      }
      [data-testid="stVerticalBlock"] > div:has([data-testid="stChatMessage"])::-webkit-scrollbar-track {
        background: transparent;
      }
      [data-testid="stVerticalBlock"] > div:has([data-testid="stChatMessage"])::-webkit-scrollbar-thumb {
        background: rgba(167,139,250,0.4);
        border-radius: 99px;
      }

      /* ── POINT 7: Glowing circular send button ── */
      [data-testid="stChatInputSubmitButton"] {
        display: flex !important;
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: 0 0 12px rgba(124,58,237,0.7), 0 0 24px rgba(124,58,237,0.4) !important;
        transition: all 0.2s ease !important;
        margin-right: 4px;
      }
      [data-testid="stChatInputSubmitButton"]:hover {
        box-shadow: 0 0 20px rgba(167,139,250,0.9), 0 0 40px rgba(167,139,250,0.5) !important;
        transform: scale(1.05) !important;
      }
      [data-testid="stChatInputSubmitButton"] svg {
        fill: white !important;
        stroke: white !important;
        color: white !important;
      }

      /* ── Chat input — remove red border, purple focus only ── */
      [data-testid="stChatInput"] textarea,
      [data-testid="stChatInput"] > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
        font-family: 'Outfit', sans-serif !important;
        outline: none !important;
        box-shadow: none !important;
      }
      [data-testid="stChatInput"] textarea:focus,
      [data-testid="stChatInput"]:focus-within > div {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 2px rgba(167,139,250,0.2) !important;
      }
      /* Remove any red Streamlit error/focus ring everywhere */
      * { outline-color: #a78bfa !important; }
      *:focus { box-shadow: none !important; }

      /* ── POINT 3: Typing animation dots ── */
      @keyframes blink {
        0%, 80%, 100% { opacity: 0; transform: scale(0.7); }
        40% { opacity: 1; transform: scale(1); }
      }
      .typing-indicator {
        display: flex; align-items: center; gap: 5px; padding: 8px 14px;
        background: rgba(255,255,255,0.06); border: 1px solid rgba(96,165,250,0.2);
        border-radius: 18px 18px 18px 4px; width: fit-content; margin-bottom: 0.6rem;
      }
      .typing-indicator span {
        width: 8px; height: 8px; background: #a78bfa;
        border-radius: 50%; display: inline-block; animation: blink 1.4s infinite;
      }
      .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
      .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

      /* ── POINT 5: Empty state ── */
      .empty-state {
        text-align: center; padding: 3rem 1rem;
        color: #475569;
      }
      .empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
      .empty-state h3 { color: #64748b; font-weight: 600; margin-bottom: 0.4rem; }
      .empty-state p { color: #475569; font-size: 0.9rem; }

      /* ── Timestamp ── */
      .msg-time { font-size: 0.7rem; color: #475569; margin-top: 2px; }

      /* ── General buttons ── */
      .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
      }
      .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(124,58,237,0.5) !important;
      }

      /* ── File uploader ── */
      [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1.5px dashed rgba(167,139,250,0.4) !important;
        border-radius: 12px !important; padding: 0.5rem !important;
      }

      hr { border-color: rgba(255,255,255,0.08) !important; }
      footer { visibility: hidden; }
    </style>
    """)


def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def process_uploaded_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    if st.session_state.get("last_processed_hash") == file_hash:
        st.sidebar.info("This document is already loaded!")
        return

    with st.spinner(f"Processing '{uploaded_file.name}'..."):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        try:
            run_ingestion_pipeline(file_path)
            st.session_state["last_processed_hash"] = file_hash
            st.session_state["last_processed_name"] = uploaded_file.name
            st.session_state["messages"] = []
            st.sidebar.success(f"✅ Ready to chat about '{uploaded_file.name}'!")
        except Exception as e:
            st.sidebar.error(f"❌ {e}")


def chat_interface():
    # Gradient header
    st.markdown("""
    <div class="app-header">
        <h1>🧠 DocuMind AI</h1>
        <p>Upload a PDF. Ask anything. Get instant answers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Active document pill
    if st.session_state.get("last_processed_name"):
        st.markdown(
            f'<div style="text-align:center;margin-bottom:0.5rem">'
            f'<span class="doc-pill">📄 {st.session_state["last_processed_name"]}</span></div>',
            unsafe_allow_html=True
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # POINT 5: Empty state when no messages and doc loaded
    if st.session_state.get("last_processed_name") and not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💬</div>
            <h3>Your document is ready!</h3>
            <p>Ask me anything — summaries, facts, details, comparisons...</p>
        </div>
        """, unsafe_allow_html=True)
    elif not st.session_state.get("last_processed_name"):
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📂</div>
            <h3>No document loaded</h3>
            <p>Upload and process a PDF from the sidebar to start chatting.</p>
        </div>
        """, unsafe_allow_html=True)

    # Display chat messages with timestamps
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("time"):
                st.markdown(
                    f'<div class="msg-time">{message["time"]}</div>',
                    unsafe_allow_html=True
                )

    # Chat input
    if st.session_state.get("last_processed_name"):
        if prompt := st.chat_input("Ask a question about your document..."):
            now = datetime.datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": prompt, "time": now})

            with st.chat_message("user"):
                st.markdown(prompt)
                st.markdown(f'<div class="msg-time">{now}</div>', unsafe_allow_html=True)

            # POINT 3: Typing indicator before response
            typing_placeholder = st.empty()
            typing_placeholder.markdown("""
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
            """, unsafe_allow_html=True)

            query_vector = create_embeddings_for_texts([prompt])[0]
            results = query_vector_store(query_vector, n_results=3)

            retrieved_chunks = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                for i in range(len(docs)):
                    retrieved_chunks.append({"text": docs[i], "metadata": metas[i]})

            if not retrieved_chunks:
                response = "I couldn't find relevant context. Please upload and process a document first."
            else:
                response = generate_answer(prompt, retrieved_chunks)

            # Clear typing indicator, show real response
            typing_placeholder.empty()

            with st.chat_message("assistant"):
                st.markdown(response)
                st.markdown(f'<div class="msg-time">{now}</div>', unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": response, "time": now})
    else:
        st.chat_input("Upload a document first...", disabled=True)


def sidebar():
    with st.sidebar:
        st.markdown("## 📂 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

        if uploaded_file is not None:
            st.markdown(f"**Selected:** `{uploaded_file.name}`")
            if st.button("⚙️ Process Document", use_container_width=True):
                process_uploaded_file(uploaded_file)

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.markdown("""
        **🔧 Pipeline**
        - Embeddings: BGE base v1.5
        - Vector DB: ChromaDB
        - LLM: Groq llama-3.1-8b
        """)


if __name__ == "__main__":
    inject_css()
    sidebar()
    chat_interface()
