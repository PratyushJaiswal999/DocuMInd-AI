import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load .env from the project root (parent of src/) regardless of CWD
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=_env_path, override=True)

# Use the explicitly requested Groq model
GROQ_MODEL = "llama-3.1-8b-instant"

def get_llm():
    """
    Initializes the Groq LLM via LangChain.
    Expects GROQ_API_KEY to be set in the environment variables.
    """
    print(f"Connecting to Groq API (model: {GROQ_MODEL})...")
    
    if "GROQ_API_KEY" not in os.environ:
        print("WARNING: GROQ_API_KEY environment variable not found.")
        print("Please set it before running the script (e.g. `$env:GROQ_API_KEY='your_key'`)")
        
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.environ.get("GROQ_API_KEY", ""),
        temperature=0, # 0 for more factual/deterministic answers
    )
    return llm

def build_rag_chain():
    """
    Builds the generation chain using prompt | llm | output_parser.
    """
    llm = get_llm()
    
    prompt_template = """\
You are a helpful and precise assistant. Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Correct LCEL pattern: prompt receives a dict with 'context' and 'question' keys
    chain = prompt | llm | StrOutputParser()
    
    return chain

def generate_answer(question: str, context_chunks: list[dict]):
    """
    Given a question and retrieved chunks, generates the final answer.
    """
    chain = build_rag_chain()
    
    # Combine the retrieved text chunks into one single context string.
    context_text = "\n\n".join([chunk.get("text", "") for chunk in context_chunks])
    print(f"Context length: {len(context_text)} chars")
    
    print(f"\n--- Generating Answer for: '{question}' ---")
    response = chain.invoke({
        "context": context_text,
        "question": question
    })
    
    return response

if __name__ == "__main__":
    print("Inference module ready.")
