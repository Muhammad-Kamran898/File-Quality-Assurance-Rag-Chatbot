# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A File QA RAG (Retrieval-Augmented Generation) Chatbot built with Streamlit, LangChain, and Ollama (free local LLMs). Users can upload PDF documents and ask questions about their content. The app uses semantic search to retrieve relevant document chunks and generates answers using local Ollama models.

## Commands

```bash
# Run the Streamlit app
streamlit run app.py

# Install dependencies (using uv)
uv sync

# Run with custom port
streamlit run app.py --server.port 8501
```

## Prerequisites

Before running the app, ensure Ollama is installed and running with the required models:

```bash
# Pull the LLM model
ollama pull llama3.2

# Pull the embedding model
ollama pull nomic-embed-text
```

## Architecture

The application uses a history-aware RAG chain with streaming responses:

```
app.py
├── configure_retriever(uploaded_files)
│   ├── PyMuPDFLoader - loads PDF files
│   ├── RecursiveCharacterTextSplitter - splits into 1500-char chunks (200 overlap)
│   ├── OllamaEmbeddings (nomic-embed-text) - creates vector embeddings
│   └── Chroma - vector database storage + retriever
│
├── StreamHandler - streams LLM response tokens to UI in real-time
│
├── PostMessageHandler - captures source documents after retrieval
│
└── qa_rag_chain (LangChain LCEL)
    ├── create_history_aware_retriever - reformulates questions using chat history
    │   └── contextualize_q_prompt - system prompt for question reformulation
    │
    ├── create_retrieval_chain - combines history-aware retriever with QA chain
    │   └── create_stuff_documents_chain - streams context + question to LLM
    │       └── qa_prompt - system prompt for answering questions
    │
    └── ChatOllama (llama3.2) - generates responses
```

**Data Flow:**
1. User uploads PDFs in sidebar
2. Documents are loaded, chunked, embedded (nomic-embed-text), and stored in ChromaDB
3. User enters a question
4. History-aware retriever reformulates question using chat context
5. Retriever finds relevant document chunks
6. Chunks + question → LLM (llama3.2) → response (streamed token-by-token)
7. Source citations displayed below answer

## Environment Variables

- `OLLAMA_BASE_URL` - Local Ollama URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - LLM model for text generation (default: llama3.2)
- `OLLAMA_EMBED_MODEL` - Embedding model (default: nomic-embed-text)

## Key Implementation Details

**Streaming:** Uses custom `StreamHandler` callback with `get_script_run_ctx()` and `add_script_run_ctx()` to stream tokens from background thread to Streamlit UI.

**History-Aware Retrieval:** Uses LangChain's `create_history_aware_retriever` to reformulate questions that reference chat history into standalone questions before retrieval.

**Source Attribution:** Custom `PostMessageHandler` captures retrieved documents and displays top 3 sources in a DataFrame after generation completes.

**Persistence:** ChromaDB persists to `./chroma_db_store` directory. If no new files are uploaded, loads existing database.

## Key Files

- `app.py` - Main application (single file with all logic)
- `pyproject.toml` - Dependencies: streamlit, langchain, chromadb, langchain-community, pymupdf
- `.env` - Ollama configuration
- `chroma_db_store/` - Persisted vector database (created at runtime)