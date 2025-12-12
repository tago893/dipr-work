# ArXiv RAG System 📚

A Retrieval-Augmented Generation (RAG) system for querying academic papers from the ArXiv dataset. This system uses Milvus for vector storage, SPECTER for embeddings, and supports multiple LLM providers (Gemini, OpenAI, HuggingFace) via a unified service interface.

## 🏗️ Architecture

### **High-Level Flow**

```text

Query Flow:
+------------+       +-------------------+       +----------------+
| User Query | ----> | Embedding Service | ----> | Milvus Service |
+------------+       +-------------------+       +-------+--------+
                                                         |
                                                         | Search (Cosine)
                                                         v
                                                +--------------------+
                                                | Semantic Retriever |
                                                +--------+-----------+
                                                         |
                                                         | Top-k Indices
                                                         v
   +--------------+      Lookup Context         +-------------+
   | Metadata CSV | <-------------------------> | LLM Service |
   +--------------+                             +------+------+
                                                       |
                                                       | Prompt + Context
                                                       v
                                               +----------------+
                                               |  LLM Provider  |
                                               +-------+--------+
                                                       |
                                                       | Response
                                                       v
                                                    +------+
                                                    | User |
                                                    +------+
```

### **Core Services (`src/`)**

| Service | File Path | Role |
|---------|-----------|------|
| **LLM Service** | `src/generation/llm_service.py` | **Unified Interface** for LLMs. Handles context building, prompt construction, and dispatching to Gemini/OpenAI/HF clients. |
| **RAG Pipeline** | `src/generation/milvus_rag_pipeline.py` | **Orchestrator**. Connects the Retriever, Embedder, and LLM Service to execute the RAG flow. |
| **Milvus Service** | `src/vector_db/milvus_service.py` | **Vector DB**. Manages the Milvus database, handles schema, indexing, and vector insertion/search. |
| **Retriever** | `src/retrieval/retriever.py` | **Search Logic**. Uses Milvus Client to find relevant document indices based on query embeddings. |
| **Embedder** | `src/embedding/embedding_service.py` | **Vectorization**. Uses `allenai-specter` to convert text queries and documents into 768-dim vectors. |

---

## 📂 Project Structure

```
├── main.py                     # 🚀 Main entry point (CLI / Interactive)
├── prepare_arxiv_data.py       # 🛠️ Step 1: Data Prep Script
├── generate_embeddings.py      # 🛠️ Step 2: Embedding Script
├── data/
│   ├── raw/                    # Place arxiv-metadata-oai-snapshot.json here
│   ├── csv_files/              # Generated CSV chunks (for processing)
│   ├── processed/              # metadata_all.csv (for runtime lookup)
│   ├── embeddings/             # Generated .npy embedding files
│   └── milvus/                 # Milvus database file (sqlite/local)
└── src/
    ├── data_ingestion/         # Data ingestion to Milvus
    ├── generation/             # LLM logic & Pipeline
    ├── retrieval/              # Search logic
    ├── vector_db/              # Database connection
    └── embedding/              # Model loading
```

---

## ⚡ Setup Guide (From Scratch)

Follow these steps in order to set up the system.

### 1. Prerequisites
- Python 3.9+
- [ArXiv Metadata Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) (download `arxiv-metadata-oai-snapshot.json`)
- API Keys for one or more providers:
    - Google Gemini (`GEMINI_API_KEY`)
    - OpenAI (`OPENAI_API_KEY`)
    - Hugging Face (`HUGGINGFACE_API_KEY`)

### 2. Installation
```bash
# Create virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```ini
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

### 4. Prepare ArXiv Data
Note: Please do remove hi files in data folder and its subfolders, they serve no purpose in the codebase
Convert the raw JSON into processed CSVs:
```bash
# Place arxiv-metadata-oai-snapshot.json in data/raw/ first!
python prepare_arxiv_data.py --json-path data/raw/arxiv-metadata-oai-snapshot.json --chunk-size 80000
```
**Output:** 
- `data/csv_files/metadata_*.csv` - Chunks for embedding generation
- `data/processed/metadata_all.csv` - Combined file for runtime lookup

### 5. Generate Embeddings
Convert papers into 768-dim vectors using SPECTER:
```bash
python generate_embeddings.py --csv data/csv_files/metadata_0_80000.csv
```
**Output:** `data/embeddings/embeddings_0_80000.npy`

>  Takes ~30-60 min on T4 GPU for 80k papers on google colab. Repeat for additional chunks if needed.

### 6. Ingest Data into Milvus
Load embeddings into the vector database:
```bash
python -c "from src.data_ingestion.ingest_to_milvus import ingest_to_milvus; ingest_to_milvus('data/embeddings/embeddings_0_80000.npy', 'data/csv_files/metadata_0_80000.csv', 'data/milvus/arxiv_papers.db', 'arxiv_papers')"
```
**OR** run `main.py` and it will auto-ingest on first run if data is missing.

### 7. Run the Application 🚀
```bash
# Interactive mode (default)
python main.py

# Specify LLM provider
python main.py --provider openai

# Skip data checks for faster startup
python main.py --skip-check
```

Ask questions like:
```
💬 What are transformer models?
💬 Explain attention mechanisms
```

---

## 🛠️ Unified LLM Service
The system uses a factory pattern in `src/generation/llm_service.py` to switch between providers seamlessly.

**Usage:**
```python
# Generate
response = llm.generate_response(query, context_documents)
print(response.content)
```

---

## 🔮 Future Roadmap

As I continue to develop this project, I have identified several key areas for improvement and feature expansion. Here is what I plan to work on next:

### **1. 🔐 Role-Based Access Control (RBAC)**
I plan to implement secure access management using [Milvus's native RBAC capabilities](https://milvus.io/docs/rbac.md). This will allow me to define granular roles and permissions directly within the vector database, ensuring robust data security for multi-user environments.

### **2. 🎯 Metadata Filtering with Categories**
I will incorporate the ArXiv `categories` field into the Milvus schema to enable hybrid search. This allows filtering by specific scientific domains (e.g., *Quantum Physics*) before semantic retrieval, significantly improving result relevance and precision.

### **3. ⚡ Scalability & Optimization**
I intend to address current memory bottlenecks by refactoring `ingest_to_milvus.py` to use optimized column loading instead of reading the full 6GB CSV. Additionally, I will perform a general codebase cleanup to remove unused utilities and streamline logic for better maintainability.

