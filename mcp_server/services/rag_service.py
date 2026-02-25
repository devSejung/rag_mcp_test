from typing import List, Dict, Any, Optional
from shared.repositories.vector_store import VectorRepository
from shared.apis.openai_client import ai_client
from shared.apis.sparse_embedder import sparse_embedder
import asyncio

class RAGService:
    """Service layer for RAG operations."""
    
    def __init__(self, repository: VectorRepository):
        self.repository = repository

    async def retrieve_and_rerank(self, query: str, top_k: int = 5, filter_component: str = None) -> List[Dict[str, Any]]:
        """
        1. Get embedding for the query.
        2. Search the vector store (Child chunks).
        3. Retrieve Parent documents.
        4. Rerank the Parent documents for better precision.
        """
        # 1. Embeddings (Dense + Sparse)
        query_vector = (await ai_client.get_embeddings([query]))[0]
        sparse_indices, sparse_values = sparse_embedder.generate_sparse_vectors([query])
        
        # 2. Search (Retrieval) - Fetch more children to find diverse parents using Hybrid Search
        limit = top_k * 3
        
        dense_task = self.repository.search_dense(query_vector, limit=limit, filter_component=filter_component)
        sparse_task = self.repository.search_sparse(sparse_indices[0], sparse_values[0], limit=limit, filter_component=filter_component)
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        scores: Dict[Any, float] = {}
        payloads: Dict[Any, Dict[str, Any]] = {}
        
        for rank, res in enumerate(dense_results, start=1):
            doc_id = res['id']
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            payloads[doc_id] = res['payload']
            
        for rank, res in enumerate(sparse_results, start=1):
            doc_id = res['id']
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            payloads[doc_id] = res['payload']
            
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        search_results = [{"id": doc_id, "score": score, "payload": payloads[doc_id]} for doc_id, score in sorted_scores]
        
        if not search_results:
            return []

        # 3. Retrieve Parents
        parent_keys = [res["payload"].get("parent_key", res["payload"].get("key")) 
                       for res in search_results 
                       if res["payload"].get("parent_key") or res["payload"].get("key")]
        
        parent_docs_payloads = await self.repository.get_parents_by_keys(parent_keys)
        
        # Fallback if no parents found (e.g. using older single-chunk data)
        if not parent_docs_payloads:
            retrieved_items = [res["payload"] for res in search_results]
        else:
            retrieved_items = parent_docs_payloads

        # 4. Reranking using full text
        docs = [item.get("full_formatted_text", item.get("content", "")) for item in retrieved_items]
        
        if not docs:
            return []
            
        reranked_indices = await ai_client.rerank(query, docs, top_n=top_k)
        
        # Map back to final results
        final_results = []
        for item in reranked_indices:
            idx = item["index"]
            original_payload = retrieved_items[idx]
            final_results.append({
                "payload": original_payload,
                "relevance_score": item["relevance_score"]
            })
            
        return final_results
