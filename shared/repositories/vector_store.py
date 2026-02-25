from qdrant_client import QdrantClient, models
from typing import List, Dict, Any, Optional
from shared.config import settings
import uuid
import logging

logger = logging.getLogger(__name__)

class VectorRepository:
    """Repository for interacting with Qdrant Vector Store."""
    
    _client_instance: Optional[QdrantClient] = None
    
    def __init__(self, collection_name: str):
        if VectorRepository._client_instance is None:
            if settings.QDRANT_URL:
                VectorRepository._client_instance = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
            else:
                VectorRepository._client_instance = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
        
        self.client = VectorRepository._client_instance
        self.collection_name = collection_name
        self.dense_dim = 1024 # Assumed BAAI dimension, adjust if necessary
        
    def initialize_collection(self):
        """Drops and recreates collection with Hybrid Search schema (Dense + Sparse)."""
        logger.warning(f"Recreating collection {self.collection_name} for Hybrid Search Schema!!!")
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"dense": models.VectorParams(size=self.dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            }
        )
        logger.info(f"Collection {self.collection_name} ready.")

    def search_dense(self, dense_vector: List[float], limit: int = 10, filter_component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search using Dense vectors."""
        qdrant_filter = None
        if filter_component:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="components",
                        match=models.MatchValue(value=filter_component)
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", dense_vector),
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]

    def search_sparse(self, sparse_indices: List[int], sparse_values: List[float], limit: int = 10, filter_component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search using Sparse vectors (BM25 custom IDF)."""
        qdrant_filter = None
        if filter_component:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="components",
                        match=models.MatchValue(value=filter_component)
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(
                name="sparse",
                vector=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            ),
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]

    def upsert_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]], sparse_indices: Optional[List[List[int]]] = None, sparse_values: Optional[List[List[float]]] = None):
        """Upsert documents with their named vectors."""
        points = []
        for i, doc in enumerate(documents):
            # Create a deterministic UUID so upserts overwrite appropriately
            doc_id_str = str(doc.get("doc_id", uuid.uuid4()))
            deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id_str))
            
            vector_data = {"dense": vectors[i]}
            if sparse_indices and sparse_values:
                vector_data["sparse"] = models.SparseVector(
                    indices=sparse_indices[i],
                    values=sparse_values[i]
                )
                
            points.append(
                models.PointStruct(
                    id=deterministic_id,
                    vector=vector_data,
                    payload=doc
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def upsert_parent_child(self, parent_doc: Dict[str, Any], child_documents: List[Dict[str, Any]], child_vectors: List[List[float]], child_sparse_indices: Optional[List[List[int]]] = None, child_sparse_values: Optional[List[List[float]]] = None):
        """
        Upsert a parent document and its child chunks using named vectors.
        """
        points = []
        
        # 1. Add Parent
        parent_id_str = str(parent_doc["key"]) # e.g. "PROJ-123"
        parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "parent_" + parent_id_str))
        
        # Named vectors config
        parent_vector = {"dense": [0.0] * self.dense_dim}
        
        parent_payload = parent_doc.copy()
        parent_payload["_type"] = "parent"
        
        points.append(
            models.PointStruct(
                id=parent_uuid,
                vector=parent_vector,
                payload=parent_payload
            )
        )
        
        # 2. Add Children
        for i, doc in enumerate(child_documents):
            doc_id_str = str(doc.get("doc_id", uuid.uuid4()))
            child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, "child_" + doc_id_str))
            
            doc_payload = doc.copy()
            doc_payload["_type"] = "child"
            doc_payload["parent_key"] = parent_id_str # Reference link
            
            vector_data = {"dense": child_vectors[i]}
            if child_sparse_indices and child_sparse_values:
                vector_data["sparse"] = models.SparseVector(
                    indices=child_sparse_indices[i],
                    values=child_sparse_values[i]
                )
            
            points.append(
                models.PointStruct(
                    id=child_uuid,
                    vector=vector_data,
                    payload=doc_payload
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def get_parents_by_keys(self, parent_keys: List[str]) -> List[Dict[str, Any]]:
        """Retrieve full parent documents sequentially based on their original keys."""
        # Convert string keys to their deterministic parent UUIDs
        parent_uuids = [str(uuid.uuid5(uuid.NAMESPACE_URL, "parent_" + key)) for key in parent_keys]
        
        # We uniquely filter UUIDs
        parent_uuids = list(set(parent_uuids))
        if not parent_uuids:
            return []
            
        # Retrieve from Qdrant by IDs
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.HasIdCondition(has_id=parent_uuids)
                ]
            ),
            limit=len(parent_uuids),
            with_payload=True,
            with_vectors=False
        )
        
        return [record.payload for record in records if record.payload]

    def get_all_keys(self, source: str) -> set:
        """Retrieve all parent keys for a specific source to perform deletion scrub."""
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source)
                    ),
                    models.FieldCondition(
                        key="_type",
                        match=models.MatchValue(value="parent")
                    )
                ]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        return {record.payload.get("key") for record in records if record.payload and record.payload.get("key")}

    def delete_by_keys(self, keys: List[str]):
        """Delete complete parent and child chunks by parent original keys."""
        if not keys:
            return
            
        parent_uuids = [str(uuid.uuid5(uuid.NAMESPACE_URL, "parent_" + key)) for key in keys]
        
        # 1. Delete Parents
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=parent_uuids
            )
        )
        
        # 2. Delete Children
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="parent_key",
                        match=models.MatchAny(any=keys)
                    )
                ]
            )
        )

# Concrete implementations
class JiraRepository(VectorRepository):
    def __init__(self):
        super().__init__(settings.JIRA_COLLECTION)

class ConfluenceRepository(VectorRepository):
    def __init__(self):
        super().__init__(settings.CONFLUENCE_COLLECTION)
