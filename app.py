import asyncio
import streamlit as st
import os
import threading
from typing import Any
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from vllm import LLM
from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel

# Set page config
st.set_page_config(page_title="Offline RAG Assistant", page_icon="🤖")

# Background event loop for all async operations
# This avoids conflicts with Streamlit's own event loop and uvloop issues
if "async_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    st.session_state.async_loop = loop
    st.session_state.async_thread = thread

def run_async(coro):
    """Run a coroutine in the background event loop and return the result."""
    future = asyncio.run_coroutine_threadsafe(coro, st.session_state.async_loop)
    return future.result()

st.title("🤖 Offline RAG Assistant")
st.markdown("""
This app demonstrates a RAG (Retrieval Augmented Generation) system using:
- **LlamaIndex**: For document indexing and retrieval
- **Milvus**: As vector store backend
- **vLLM**: For offline text generation
- **Pydantic AI**: For agent-based query processing
""")

# Sidebar configuration
st.sidebar.header("Configuration")
data_dir = st.sidebar.text_input("Data Directory", value="./data")
embedding_model_name = st.sidebar.text_input("Embedding Model", value="BAAI/bge-small-en-v1.5")
chat_model_name = st.sidebar.text_input("Chat Model", value="microsoft/Phi-4-mini-instruct")
db_path = st.sidebar.text_input("Milvus DB Path", value="./milvus_demo.db")
gpu_utilization = st.sidebar.slider("GPU Memory Utilization", 0.1, 0.9, 0.8)
max_num_seqs = st.sidebar.number_input("Max Num Seqs", value=32)
chunk_size = st.sidebar.number_input("Chunk Size", value=1000)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=200)
top_k = st.sidebar.slider("Top K Retrieval", 1, 10, 3)

@st.cache_resource
def get_models(embedding_model_name, chat_model_name, chunk_size, chunk_overlap, gpu_utilization, max_num_seqs):
    try:
        with st.status("Initializing models... (This may take a while)") as status:
            # Initialize offline embedding model
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model_name
            )

            # Initialize vLLM offline
            vllm_model = LLM(
                model=chat_model_name,
                gpu_memory_utilization=gpu_utilization,
                max_num_seqs=max_num_seqs,
                max_model_len=1616
            )

            # Wrap with OutlinesModel
            model = OutlinesModel.from_vllm_offline(vllm_model)

            Settings.transformations = [
                SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            ]
            status.update(label="Models initialized!", state="complete")
            return model
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return None

@st.cache_resource
def get_index(_model, data_dir, db_path):
    if not os.path.exists(data_dir):
        st.error(f"Data directory '{data_dir}' does not exist.")
        return None
    
    async def _create_index():
        documents = SimpleDirectoryReader(data_dir).load_data()
        if not documents:
            return None
            
        sample_emb = Settings.embed_model.get_text_embedding("test")
        vector_store = MilvusVectorStore(uri=db_path, dim=len(sample_emb), overwrite=True)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        return index

    try:
        with st.status("Loading documents and creating index...") as status:
            index = run_async(_create_index())
            status.update(label="Index created!", state="complete")
            return index
    except Exception as e:
        st.error(f"Error creating index: {e}")
        return None

async def query_document(index, question, top_k, model):
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

# Initialize models and index
if st.sidebar.button("Reload Index & Models"):
    st.cache_resource.clear()
    st.rerun()

model = get_models(embedding_model_name, chat_model_name, chunk_size, chunk_overlap, gpu_utilization, max_num_seqs)

if model:
    index = get_index(model, data_dir, db_path)

    if index:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about your documents"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Run async query in the background loop
                        response = run_async(query_document(index, prompt, top_k, model))
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error during query: {e}")
    else:
        st.info("Please ensure the data directory contains documents and click 'Reload Index & Models' if needed.")
else:
    st.info("Model initialization failed. Check the configuration and ensure you have enough GPU memory.")
