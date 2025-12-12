"""Simplified RAG Pipeline using Milvus semantic search."""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9


@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    content: str
    confidence: float
    generation_time: float
    token_count: int


@dataclass
class RAGResult:
    """Complete RAG result."""
    query: str
    generated_response: GeneratedResponse
    retrieved_documents: List[Dict[str, Any]]
    retrieval_time: float
    total_time: float
    num_documents_used: int


class MilvusRAGPipeline:
    """RAG pipeline using Milvus semantic search."""
    
    def __init__(self, retriever, embedding_service, llm_service):
        self.retriever = retriever
        self.embedding_service = embedding_service
        self.llm_service = llm_service
    
    def query(self, question: str, top_k: int = 2, generation_config: Optional[GenerationConfig] = None) -> RAGResult:
        """Execute RAG query."""
        start_time = time.time()
        
        query_embedding = self.embedding_service.embed_query(question)[0]
        retrieved_docs = self.retriever.search(query_embedding, top_k)
        retrieval_time = time.time() - start_time
        
        if self.llm_service and retrieved_docs:
            generated_response = self.llm_service.generate_response(question, retrieved_docs, generation_config)
        else:
            generated_response = GeneratedResponse(
                content="Error: No documents retrieved or LLM service not configured.",
                confidence=0.0, generation_time=0.0, token_count=0
            )
        
        return RAGResult(
            query=question,
            generated_response=generated_response,
            retrieved_documents=retrieved_docs,
            retrieval_time=retrieval_time,
            total_time=time.time() - start_time,
            num_documents_used=len(retrieved_docs)
        )