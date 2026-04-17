# AI Chatbot Using RAG

This project is a simple Retrieval-Augmented Generation (RAG) chatbot built with:

- Streamlit for the web interface
- ChromaDB as the vector database
- Ollama for local LLM and embedding models
- Python for document ingestion and retrieval logic

The app lets you:

- upload `pdf`, `txt`, and `md` files
- split documents into chunks
- generate embeddings and store them in ChromaDB
- ask questions grounded in your uploaded documents
- view retrieved source chunks in the UI

## Project Structure

```text
.
├── app.py
├── rag
│   ├── config.py
│   ├── document_loader.py
│   ├── ollama_client.py
│   ├── rag_pipeline.py
│   ├── text_splitter.py
│   ├── ui.py
│   └── vector_store.py
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Ollama and pull the required models:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## How It Works

1. Upload one or more files from the sidebar.
2. Click `Ingest documents` to chunk and index them into `vector_store/`.
3. Ask questions in the chat box.
4. The app retrieves the most relevant chunks and sends them to the chat model as context.

## Notes

- The vector database is persisted locally in `vector_store/`.
- You can change the Ollama model names from the sidebar.
- If Ollama is not running, the app will show a helpful error message.
