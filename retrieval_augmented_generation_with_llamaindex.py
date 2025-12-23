# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RAG (Retrieval Augmented Generation) Implementation with LlamaIndex and Pydantic AI
================================================================

This script demonstrates a RAG system using:
- LlamaIndex: For document indexing and retrieval
- Milvus: As vector store backend
- vLLM: For offline text generation
- Pydantic AI: For agent-based query processing with Outlines

Features:
1. Document Loading & Processing
2. Embedding & Storage
3. Query Processing using Pydantic AI Agent

Requirements:
1. Install dependencies:
pip install llama-index llama-index-readers-web \
            llama-index-embeddings-huggingface \
            llama-index-vector-stores-milvus \
            vllm pydantic-ai

Usage:
    python retrieval_augmented_generation_with_llamaindex.py

Notes:
    - All models run locally (offline)
    - First run may take time to download models
"""

import argparse
import asyncio
from argparse import Namespace
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from vllm import LLM
from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel


def init_config(args: Namespace):
    """Initialize configuration with command line arguments"""
    return {
        "data_dir": args.data_dir,
        "embedding_model": args.embedding_model,
        "chat_model": args.chat_model,
        "vllm_api_key": args.vllm_api_key,
        "db_path": args.db_path,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
    }


def load_documents(data_dir: str) -> list:
    """Load and process local documents"""
    return SimpleDirectoryReader(data_dir).load_data()


def setup_models(config: dict[str, Any]):
    """Configure embedding and chat models"""
    # Initialize offline embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config["embedding_model"]
    )

    # Initialize vLLM offline
    vllm_model = LLM(
        model=config["chat_model"],
        gpu_memory_utilization=0.9,
        max_model_len=1616
    )

    # Wrap with OutlinesModel
    model = OutlinesModel.from_vllm_offline(vllm_model)

    Settings.transformations = [
        SentenceSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
    ]
    return model


def setup_vector_store(db_path: str) -> MilvusVectorStore:
    """Initialize vector store"""
    sample_emb = Settings.embed_model.get_text_embedding("test")
    print(f"Embedding dimension: {len(sample_emb)}")
    return MilvusVectorStore(uri=db_path, dim=len(sample_emb), overwrite=True)


def create_index(documents: list, vector_store: MilvusVectorStore):
    """Create document index"""
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )


async def query_document(index: VectorStoreIndex, question: str, top_k: int, model: OutlinesModel):
    """Query document with given question using pydantic-ai"""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = await retriever.aretrieve(question)
    context = "\n\n".join([n.get_content() for n in nodes])

    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant. Use the provided context to answer the user's question.",
        model_settings={
            "temperature": 0.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
    )

    result = await agent.run(f"Context:\n{context}\n\nQuestion: {question}")
    return result.output


def get_parser() -> argparse.ArgumentParser:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Offline RAG with vLLM and LlamaIndex")

    # Add command line arguments
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing documents to process",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Model name for embeddings",
    )
    parser.add_argument(
        "--chat-model", default="microsoft/Phi-4-mini-instruct", help="Model name for chat"
    )
    parser.add_argument(
        "--vllm-api-key", default="EMPTY", help="API key for vLLM (not used in offline mode)"
    )
    parser.add_argument(
        "--db-path", default="./milvus_demo.db", help="Path to Milvus database"
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enable interactive Q&A mode"
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for document splitting",
    )
    parser.add_argument(
        "-o",
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for document splitting",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=3, help="Number of top results to retrieve"
    )

    return parser


async def main():
    # Parse command line arguments
    args = get_parser().parse_args()

    # Initialize configuration
    config = init_config(args)

    # Load documents
    documents = load_documents(config["data_dir"])

    # Setup models
    model = setup_models(config)

    # Setup vector store
    vector_store = setup_vector_store(config["db_path"])

    # Create index
    index = create_index(documents, vector_store)

    if args.interactive:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            # Get user question
            question = input("\nEnter your question: ")

            # Check for exit command
            if question.lower() in ["quit", "exit", "q"]:
                print("Exiting interactive mode...")
                break

            # Get and print response
            print("\n" + "-" * 50)
            print("Response:\n")
            response = await query_document(index, question, config["top_k"], model)
            print(response)
            print("-" * 50)
    else:
        # Single query mode
        question = "Who built the Eiffel Tower and when?"
        response = await query_document(index, question, config["top_k"], model)
        print("-" * 50)
        print("Response:\n")
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())