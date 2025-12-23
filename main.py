from vllm import LLM
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.outlines import OutlinesModel
from rag_engine import RAGEngine
import os

def main():
    # Initialize RAG Engine
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    rag_engine = RAGEngine(data_dir)

    # Initialize vLLM
    vllm_model = LLM(
        'microsoft/Phi-4-mini-instruct',
        gpu_memory_utilization=0.9,
        max_model_len=1616
    )

    # Wrap with OutlinesModel
    model = OutlinesModel.from_vllm_offline(vllm_model)

    # Create the agent
    agent = Agent(
        model=model,
        system_prompt="You are a helpful assistant. Use the provided context to answer the user's question."
    )

    try:
        user_query = "Who built the Eiffel Tower and when?"
        print(f"\nQuery: {user_query}")
        
        # 1. Retrieve context
        print("Retrieving context...")
        context_chunks = rag_engine.search(user_query)
        context = "\n\n".join(context_chunks)
        
        # 2. Construct augmented prompt
        augmented_prompt = f"""Context:
{context}

Question: {user_query}
"""
        
        # 3. Run agent
        print("Generating response...")
        result = agent.run_sync(
            augmented_prompt,
            model_settings=ModelSettings(
                temperature=0.0,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
        )
        
        print(f"\nResponse: {result.output}")
        
    finally:
        if hasattr(vllm_model, 'llm_engine') and hasattr(vllm_model.llm_engine, 'engine_core'):
            vllm_model.llm_engine.engine_core.shutdown()

if __name__ == "__main__":
    main()