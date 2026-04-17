from __future__ import annotations

import streamlit as st
from requests import RequestException

from rag.config import DEFAULT_EMBED_MODEL, DEFAULT_LLM_MODEL
from rag.rag_pipeline import RAGPipeline


def run_app() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("AI Chatbot with RAG, ChromaDB, and Streamlit")
    st.caption("Upload documents, store embeddings in a vector DB, and chat over your data.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    pipeline = RAGPipeline()

    with st.sidebar:
        st.header("Settings")
        llm_model = st.text_input("Ollama chat model", value=DEFAULT_LLM_MODEL)
        embedding_model = st.text_input(
            "Ollama embedding model",
            value=DEFAULT_EMBED_MODEL,
        )
        st.metric("Stored chunks", pipeline.vector_store.count())
        st.markdown(
            "Make sure Ollama is running locally with models like "
            f"`{DEFAULT_LLM_MODEL}` and `{DEFAULT_EMBED_MODEL}` pulled."
        )

        uploaded_files = st.file_uploader(
            "Upload knowledge files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Ingest documents", use_container_width=True):
            _ingest_files(pipeline, uploaded_files, embedding_model)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your uploaded documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    answer, matches = pipeline.ask(
                        question=question,
                        llm_model=llm_model,
                        embedding_model=embedding_model,
                    )
                except RequestException as exc:
                    answer = (
                        "I could not reach the local Ollama server. "
                        "Start Ollama and confirm the selected models are installed.\n\n"
                        f"Error: `{exc}`"
                    )
                    matches = []

                st.markdown(answer)
                if matches:
                    st.subheader("Retrieved Context")
                    for index, match in enumerate(matches, start=1):
                        source = match["metadata"]["source"]
                        score = 1 - float(match["distance"])
                        st.markdown(
                            f"**{index}. {source}**  \n"
                            f"Similarity: `{score:.3f}`  \n"
                            f"{match['document']}"
                        )

        st.session_state.messages.append({"role": "assistant", "content": answer})


def _ingest_files(
    pipeline: RAGPipeline,
    uploaded_files: list,
    embedding_model: str,
) -> None:
    if not uploaded_files:
        st.warning("Upload at least one file before ingesting.")
        return

    inserted_chunks = 0
    processed_files = 0
    try:
        for uploaded_file in uploaded_files:
            _, chunk_count = pipeline.ingest_document(
                file_name=uploaded_file.name,
                file_bytes=uploaded_file.getvalue(),
                embedding_model=embedding_model,
            )
            inserted_chunks += chunk_count
            processed_files += 1
    except ValueError as exc:
        st.error(str(exc))
        return
    except RequestException as exc:
        st.error(
            "Could not reach the local Ollama server while generating embeddings. "
            f"Error: `{exc}`"
        )
        return

    st.success(
        f"Processed {processed_files} file(s) and stored {inserted_chunks} chunks."
    )
