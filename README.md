# 📄 Document QA System using RAG

A Retrieval-Augmented Generation (RAG) application that lets you upload any PDF
and ask natural language questions grounded in the document's content

**Stack:** Streamlit · LangChain · FAISS · HuggingFace Embeddings · Groq (Free LLM)

---

## 🚀 Quick Setup

### 1. Clone / download the project
```bash
cd rag_doc_qa
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Open .env and add your GROQ_API_KEY (free at https://console.groq.com)
```

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔑 API Keys

| Provider | Cost | Where to get |
|---|---|---|
| **Groq** ✅ Recommended | FREE | https://console.groq.com |
| OpenAI | PAID | https://platform.openai.com |
| HuggingFace | FREE (less reliable) | https://huggingface.co/settings/tokens |

---

## 🏗️ Architecture

```
PDF Upload
    ↓
PyPDFLoader  →  RecursiveCharacterTextSplitter  →  Chunks
                                                       ↓
                                            HuggingFace Embeddings
                                                       ↓
                                              FAISS Vector Store
                                                       ↑
User Query  →  Embed Query  →  FAISS Top-K Retrieval
                                        ↓
                          Prompt Template + Retrieved Context
                                        ↓
                                  Groq LLM (Free)
                                        ↓
                                  Answer + Sources
```

## ⚙️ Configuration (in app.py)

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 1000 | Token size per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `embedding_model` | all-MiniLM-L6-v2 | HuggingFace embedding model |
| `llm_provider` | groq | groq / openai / huggingface |
| `llm_model` | llama-3.3-70b-versatile | LLM model name |
| `temperature` | 0.5 | Generation temperature |
| `retrieval_k` | 3 | Number of chunks to retrieve |

---

## 📁 Project Structure

```
rag_doc_qa/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .env                # Your actual keys (DO NOT commit)
└── README.md           # This file
```
