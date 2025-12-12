"""Semantic retriever for ArXiv papers."""

from typing import List, Dict, Any, Union
import numpy as np
from vector_db.milvus_service import MilvusClient
import traceback

class SemanticRetriever:
    """Retriever that handles semantic search."""
    
    def __init__(self, milvus_client: MilvusClient):
        """Initialize retriever with Milvus client."""
        self.milvus_client = milvus_client
        
    def search(self, query_embedding: Union[List[float], np.ndarray], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents using query embedding.
        """
        try:
            # Search Milvus
            results = self.milvus_client.search(query_embedding, top_k=top_k)
            
            # Convert SearchResult objects to dicts
            documents = []
            for result in results:
                doc = {
                    'arxiv_id': result.arxiv_id,
                    'embedding_index': result.embedding_index,
                    'similarity_score': result.similarity_score,
                    'retrieval_method': 'semantic'
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            traceback.print_exc()
            return []
            
    def get_status(self) -> Dict[str, Any]:
        """Get retriever status."""
        return {
            'milvus_connected': self.milvus_client is not None
        }
