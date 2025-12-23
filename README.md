# RAG AI Sample Application

This is a sample Retrieval-Augmented Generation (RAG) application using `pydantic-ai`, `vllm`, and `sentence-transformers`.

## Features
- **Local LLM**: Uses `microsoft/Phi-4-mini-instruct` via `vllm`.
- **Vector Search**: Simple in-memory vector search using `sentence-transformers` and `numpy`.
- **RAG Pipeline**: Retrieves relevant context from local text files and uses it to answer user queries.

## Project Structure
- `main.py`: The main entry point of the application.
- `rag_engine.py`: Contains the `RAGEngine` class for document embedding and retrieval.
- `data/`: Directory containing sample knowledge base text files.

## Setup and Usage

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Run the Application**:
   ```bash
   uv run python main.py
   ```

## How it Works
1. The `RAGEngine` loads text files from the `data/` directory.
2. It chunks the text and creates embeddings using the `all-MiniLM-L6-v2` model.
3. When a query is received, it searches for the most relevant chunks.
4. The retrieved context is prepended to the user's query.
5. The `pydantic-ai` agent generates a response using the augmented prompt.
