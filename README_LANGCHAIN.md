# Hospital Voice Nearby — LangChain + Voice

## Quick Start
```powershell
# 1) Create/activate venv (Windows example)
python -m venv .venv
. .venv\Scripts\activate

# 2) Install deps
pip install -r requirements_langchain.txt
pip install -r app\requirements_voice.txt

# 3) Build vector DB (reads data/hospitals.csv → storage/chroma/)
python -m scripts.ingest

# 4) Run the app
streamlit run app.py
```

If Chroma shows a DLL error on Windows:
```powershell
pip uninstall -y chromadb chromadb-rust-bindings
pip install chromadb==0.4.24
```

## Structure
- `app.py` — Streamlit app with Text & Voice search
- `scripts/ingest.py` — CSV → LangChain Documents → Chroma persistent store
- `app/services/lc_vector.py` — embeddings + vector store + retriever
- `app/services/retrieval.py` — wrapper for querying
- `data/hospitals.csv` — sample data (replace with yours)
- `storage/chroma/` — persisted vector DB
- `.vscode/launch.json` — run ingestion easily
