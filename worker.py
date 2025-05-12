import os, logging, torch
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ───────────────────── Logging & device ──────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────── Globals ───────────────────────────────
llm_hub = embeddings = None

def init_llm():
    """Create the HF pipeline + embeddings once."""
    global llm_hub, embeddings
    logger.info("Loading Hermes‑2‑Pro‑Mistral‑7B on %s …", DEVICE)
    pipe = pipeline(
        "text-generation",
        model="NousResearch/Hermes-2-Pro-Mistral-7B",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device=0 if DEVICE == "cuda" else -1,
        max_new_tokens=512,        # <-- bigger so answers don't truncate
        temperature=0.2,
        top_p=0.9,
    )
    llm_hub  = HuggingFacePipeline(pipeline=pipe)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
    )
    logger.info("LLM + embeddings ready.")

def process_document(path: str):
    """Load PDF, dedupe chunks, build Chroma store, return RetrievalQA chain."""
    pages = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20)
    raw_chunks = splitter.split_documents(pages)

    # Deduplicate identical text blocks
    seen, chunks = set(), []
    for doc in raw_chunks:
        text = doc.page_content.strip()
        if text not in seen:
            seen.add(text)
            chunks.append(doc)

    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory="./chroma_db"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful technical assistant.

Context:
{context}

Question:
{question}

###  Write the answer in this exact format
**Title:** <concise title>

**Summary**
- bullet 1
- bullet 2
- …

###  Answer:
""".strip(),
    )

    return RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

def process_prompt(prompt: str, qa_chain, history):
    """Ask the chain and return only the clean answer text."""
    try:
        result = qa_chain.invoke({"query": prompt})       # key must be "query"
        answer = result.get("result", "") if isinstance(result, dict) else str(result)
    except Exception as e:
        answer = f"Error: {e}"

    # Keep only text after our "Answer:" header, strip dashes/spaces
    answer = answer.split("Answer:", 1)[-1].strip().lstrip("–—").lstrip()

    history.append((prompt, answer))
    return answer
