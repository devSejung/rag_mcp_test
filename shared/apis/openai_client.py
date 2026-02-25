import httpx
from typing import List, Optional, Dict, Any
from shared.config import settings

class InternalAIClient:
    """Client for internal Embedding and Reranking APIs (OpenAI Compatible)."""
    
    def __init__(self):
        self.base_url = settings.INTERNAL_API_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {settings.INTERNAL_API_KEY}",
            "Content-Type": "application/json"
        }

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fetch embeddings from the internal API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json={
                    "input": texts,
                    "model": settings.EMBEDDING_MODEL
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI standard response format
            return [item["embedding"] for item in data["data"]]

    async def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents based on the query.
        Note: Rerank is often a custom endpoint even in 'OpenAI-like' environments.
        """
        async with httpx.AsyncClient() as client:
            # Assuming a common /rerank endpoint pattern
            response = await client.post(
                f"{self.base_url}/rerank",
                headers=self.headers,
                json={
                    "query": query,
                    "documents": documents,
                    "model": settings.RERANKER_MODEL,
                    "top_n": top_n
                },
                timeout=30.0
            )
            
            # Fallback or error handling if rerank is not available
            if response.status_code != 200:
                # Log or handle appropriately
                return [{"index": i, "relevance_score": 0.0} for i in range(len(documents))]
                
            return response.json().get("results", [])

# Singleton instance
ai_client = InternalAIClient()
