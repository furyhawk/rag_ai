import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class RAGEngine:
    def __init__(self, data_dir: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.load_documents(data_dir)

    def load_documents(self, data_dir: str):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(data_dir, filename), 'r') as f:
                    content = f.read()
                    # Simple chunking by paragraph for this example
                    chunks = [p.strip() for p in content.split('\n') if p.strip()]
                    self.documents.extend(chunks)
        
        if self.documents:
            self.embeddings = self.model.encode(self.documents)

    def search(self, query: str, top_k: int = 2) -> List[str]:
        if not self.documents:
            return []
        
        query_embedding = self.model.encode([query])
        
        # Compute cosine similarity
        # embeddings are (N, D), query_embedding is (1, D)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]

if __name__ == "__main__":
    # Quick test
    engine = RAGEngine('./data')
    results = engine.search("Who built the Eiffel Tower?")
    for res in results:
        print(f"- {res}")
