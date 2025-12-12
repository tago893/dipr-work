"""Milvus service for ArXiv papers."""

from pymilvus import MilvusClient as PyMilvusClient, DataType, FieldSchema, CollectionSchema
import numpy as np
from typing import List, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result for ArXiv papers."""
    arxiv_id: str
    embedding_index: int
    similarity_score: float


class MilvusClient:
    """Milvus client for ArXiv papers."""
    
    def __init__(self, uri: str = "data/milvus/arxiv_papers.db"):
        self.client = PyMilvusClient(uri=uri)
        self.collection_name = None
    
    def create_collection(self, collection_name: str, dimension: int = 768, drop_existing: bool = True):
        """Create collection with ArXiv schema."""
        if drop_existing and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="arxiv_id", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="embedding_index", dtype=DataType.INT64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]
        
        schema = CollectionSchema(fields=fields, description="ArXiv papers")
        self.client.create_collection(collection_name=collection_name, schema=schema)
        self.collection_name = collection_name
    
    def create_index(self, collection_name: str = None):
        """Create search index."""
        target_collection = collection_name or self.collection_name
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self.client.create_index(collection_name=target_collection, index_params=index_params)
        return True
    
    def load_collection(self, collection_name: str):
        """Load collection for operations."""
        if self.client.has_collection(collection_name):
            self.client.load_collection(collection_name)
            self.collection_name = collection_name
            return True
        return False
    
    def insert_embeddings_batch(self, embeddings: np.ndarray, arxiv_data: List[Dict[str, Any]], 
                               start_index: int = 0, batch_size: int = 1000):
        """Insert papers with embeddings."""
        if not self.collection_name:
            return False
        
        for i in range(0, len(embeddings), batch_size):
            batch_data = []
            for j in range(i, min(i + batch_size, len(embeddings))):
                batch_data.append({
                    "arxiv_id": str(arxiv_data[j].get("id", "")),
                    "embedding_index": start_index + j,
                    "vector": embeddings[j].tolist()
                })
            
            self.client.insert(collection_name=self.collection_name, data=batch_data)
        
        return True
    
    def search(self, query_vector: Union[List[float], np.ndarray], top_k: int = 5) -> List[SearchResult]:
        """Search for similar papers."""
        if not self.collection_name:
            return []
        
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["arxiv_id", "embedding_index"]
        )
        
        search_results = []
        if results and len(results) > 0:
            for hit in results[0]:
                search_results.append(SearchResult(
                    arxiv_id=hit['entity']['arxiv_id'],
                    embedding_index=hit['entity']['embedding_index'],
                    similarity_score=hit['distance']
                ))
        
        return search_results
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.collection_name:
            return {}
        
        stats = self.client.get_collection_stats(self.collection_name)
        return {
            'num_entities': int(stats.get('row_count', 0)),
            'collection_name': self.collection_name
        }