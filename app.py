"""
Document QA System using RAG
Built with Streamlit, LangChain, FAISS, HuggingFace Embeddings & Groq (Free)
"""

import streamlit as st
import os
import tempfile
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Suppress noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ─────────────────────────────────────────────
# ENV & CONFIG
# ─────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")

CONFIG = {
    "chunk_size":       1000,
    "chunk_overlap":    200,
    "embedding_model":  "sentence-transformers/all-MiniLM-L6-v2",
    "llm_provider":     "groq",               # "groq" | "openai" | "huggingface"
    "llm_model":        "llama-3.3-70b-versatile",
    "temperature":      0.5,
    "max_tokens":       512,
    "retrieval_k":      3,
}

# ─────────────────────────────────────────────
# LLM WRAPPERS
# ─────────────────────────────────────────────
from langchain_core.language_models.llms import LLM

class GroqLLM(LLM):
    """Free Groq LLM wrapper."""
    api_key:     str
    model_name:  str   = "llama-3.3-70b-versatile"
    temperature: float = 0.5
    max_tokens:  int   = 512

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except ImportError:
            raise Exception("Run: pip install groq")
        except Exception as e:
            raise Exception(f"Groq API error: {e}")


class OpenAILLM(LLM):
    """Paid OpenAI LLM wrapper."""
    api_key:     str
    model_name:  str   = "gpt-3.5-turbo"
    temperature: float = 0.5
    max_tokens:  int   = 512

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "openai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except ImportError:
            raise Exception("Run: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")


class HuggingFaceLLM(LLM):
    """Free HuggingFace Inference API wrapper."""
    model_id:    str
    hf_token:    str   = ""
    temperature: float = 0.5
    max_tokens:  int   = 512

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        try:
            import requests
            url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                },
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", str(result)).strip()
            return str(result)
        except Exception as e:
            raise Exception(f"HuggingFace API error: {e}")


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def validate_configuration() -> bool:
    provider = CONFIG["llm_provider"]
    if provider == "groq" and not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY missing. Add it to your .env file.")
        st.info("Get a FREE key at https://console.groq.com")
        return False
    if provider == "openai" and not OPENAI_API_KEY:
        st.error("❌ OPENAI_API_KEY missing.")
        return False
    if provider == "huggingface" and not HF_TOKEN:
        st.error("❌ HF_TOKEN missing.")
        return False
    return True


def process_document(uploaded_file) -> List[Document]:
    """Save upload to temp file, load with PyPDFLoader, split into chunks."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        if not pages:
            raise ValueError("PDF has no readable content.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(pages)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def create_vector_db(chunks: List[Document]) -> FAISS:
    """Embed chunks and build FAISS index."""
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embeddings)


def initialize_llm():
    provider = CONFIG["llm_provider"]
    if provider == "groq":
        return GroqLLM(
            api_key=GROQ_API_KEY,
            model_name=CONFIG["llm_model"],
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
        )
    elif provider == "openai":
        return OpenAILLM(
            api_key=OPENAI_API_KEY,
            model_name=CONFIG["llm_model"],
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
        )
    else:
        return HuggingFaceLLM(
            model_id=CONFIG["llm_model"],
            hf_token=HF_TOKEN,
            temperature=CONFIG["temperature"],
            max_tokens=CONFIG["max_tokens"],
        )


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document QA System (RAG)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    h1, h2, h3 { color: #1f77b4; }
    .stExpander { background-color: transparent !important; }
    .answer-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Document QA System (RAG)")
st.markdown("**Upload any PDF and ask intelligent questions grounded in its content.**  \n"
            "Powered by LangChain · FAISS · HuggingFace Embeddings · Groq (FREE)")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "This app uses **Retrieval-Augmented Generation (RAG)** to answer "
        "questions from your uploaded PDF.\n\n"
        "It retrieves the most relevant chunks and feeds them to an LLM."
    )

    st.header("⚙️ Configuration")
    st.json(CONFIG)

    st.header("🔑 API Status")
    c1, c2, c3 = st.columns(3)
    c1.success("✅ Groq")   if GROQ_API_KEY   else c1.warning("⬜ Groq")
    c2.success("✅ OpenAI") if OPENAI_API_KEY else c2.warning("⬜ OpenAI")
    c3.success("✅ HF")     if HF_TOKEN       else c3.warning("⬜ HF")

    st.header("🤖 Model Selection")
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "compound-beta",
    ]
    selected = st.selectbox("Groq model:", groq_models, index=0)
    CONFIG["llm_model"] = selected

    st.header("🔢 Retrieval Settings")
    CONFIG["retrieval_k"] = st.slider("Top-K chunks to retrieve:", 1, 6, CONFIG["retrieval_k"])
    CONFIG["chunk_size"]  = st.slider("Chunk size (tokens):", 200, 2000, CONFIG["chunk_size"], step=100)

# Validate keys before proceeding
if not validate_configuration():
    st.stop()

# ── File Upload ───────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload your PDF document",
    type="pdf",
    help="Any PDF — research paper, manual, report, etc.",
)

if uploaded_file:
    new_file = (
        "vector_db" not in st.session_state
        or st.session_state.get("current_file") != uploaded_file.name
    )
    if new_file:
        with st.spinner("⚙️ Processing document and building vector index…"):
            try:
                chunks = process_document(uploaded_file)
                st.session_state.vector_db    = create_vector_db(chunks)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chunk_count  = len(chunks)
                st.success(f"✅ Indexed **{len(chunks)} chunks** from `{uploaded_file.name}`")
            except Exception as e:
                st.error(f"❌ Failed to process PDF: {e}")
                st.stop()
    else:
        st.success(f"✅ Using cached index — **{st.session_state.chunk_count} chunks** ready")

    # ── Query ─────────────────────────────────
    query = st.text_input(
        "❓ Ask a question about the document:",
        placeholder="e.g. What are the main conclusions of this paper?",
    )

    if query:
        with st.spinner("🔍 Searching and generating answer…"):
            try:
                llm       = initialize_llm()
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": CONFIG["retrieval_k"]}
                )

                PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly based on the provided document context.

Rules:
- Answer ONLY using the context below.
- If the answer is not in the context, say "I couldn't find this in the document."
- Be concise, accurate, and cite relevant details from the context.

Context:
{context}

Question: {question}

Answer:"""

                prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                answer      = chain.invoke(query)
                source_docs = retriever.invoke(query)

                # Display answer
                st.subheader("💡 Answer")
                st.info(answer)
                

                # Display source chunks
                st.subheader(f"Source Chunks (Top {CONFIG['retrieval_k']} matches)")
                for i, doc in enumerate(source_docs, 1):
                    page = doc.metadata.get("page", "?")
                    with st.expander(f"Chunk {i} — Page {page}"):
                        st.info(doc.page_content)

                st.info(f"🤖 Model used: **{CONFIG['llm_model']}** | Retrieved **{len(source_docs)}** chunks")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                with st.expander("🐛 Traceback"):
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    # ── Architecture Details ──────────────────
    with st.expander("System Architecture Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Document Chunking")
            st.write(f"- **Chunk Size:** {CONFIG['chunk_size']} tokens")
            st.write(f"- **Chunk Overlap:** {CONFIG['chunk_overlap']} tokens")
            st.write("- **Splitter:** RecursiveCharacterTextSplitter")
            st.subheader("Embeddings")
            st.write(f"- **Model:** {CONFIG['embedding_model']}")
            st.write("- **Vector Store:** FAISS")
            st.write("- **Normalized:** Yes")
        with col2:
            st.subheader("LLM Configuration")
            st.write(f"- **Provider:** {CONFIG['llm_provider'].upper()}")
            st.write(f"- **Model:** {CONFIG['llm_model']}")
            st.write(f"- **Temperature:** {CONFIG['temperature']}")
            st.write(f"- **Max Tokens:** {CONFIG['max_tokens']}")
            st.write(f"- **Retrieval K:** {CONFIG['retrieval_k']}")
        st.divider()
        st.subheader("🔄 RAG Pipeline")
        steps = [
            "PDF loaded & parsed via PyPDFLoader",
            "Text split into overlapping chunks",
            "Chunks embedded with all-MiniLM-L6-v2",
            "Embeddings stored in FAISS index",
            "User query converted to embedding",
            "FAISS retrieves top-K similar chunks",
            "Chunks + query fed to LLM via prompt template",
            "LLM generates grounded answer",
            "Answer & source chunks displayed",
        ]
        for i, s in enumerate(steps, 1):
            st.write(f"{i}. {s}")

else:
    st.info("👆 Upload a PDF file above to get started.")
    st.markdown("""
### Quick Start
1. Add your `GROQ_API_KEY` to `.env` (free at https://console.groq.com)
2. Upload any PDF document
3. Ask questions — the system retrieves relevant sections and answers accurately

### Available Free Groq Models
| Model | Context | Speed |
|---|---|---|
| `llama-3.3-70b-versatile` | 128K | Fast |
| `llama-3.1-8b-instant` | 128K | Very Fast |
| `llama3-70b-8192` | 8K | Fast |
| `gemma2-9b-it` | 8K | Fast |
""")

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:gray;'>"
    f"📄 Document QA System | LangChain · FAISS · Groq | Model: <strong>{CONFIG['llm_model']}</strong>"
    f"</div>",
    unsafe_allow_html=True,
)
