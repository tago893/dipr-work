"""Unified LLM Service Factory."""

import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .gemini_client import GeminiClient, GeminiConfig
from .openai_client import OpenAIClient, OpenAIConfig
from .huggingface_client import HuggingFaceClient, HuggingFaceConfig


class LLMProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


@dataclass
class GenerationConfig:
    """Unified generation configuration."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    
    def to_gemini_config(self) -> GeminiConfig:
        return GeminiConfig(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p)
        
    def to_openai_config(self) -> OpenAIConfig:
        return OpenAIConfig(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p)

    def to_hf_config(self) -> HuggingFaceConfig:
        return HuggingFaceConfig(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p)


@dataclass
class GeneratedResponse:
    """Standardized response object."""
    content: str
    confidence: float
    generation_time: float
    token_count: int


class LLMService:
    """Unified LLM Service - abstracts provider details."""
    
    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = LLMProvider(provider.lower())
        self.metadata_df = self._load_csv()
        
        if self.provider == LLMProvider.GEMINI:
            model = model_name or "gemini-2.5-flash"
            self.client = GeminiClient(api_key=api_key, model_name=model)
        elif self.provider == LLMProvider.OPENAI:
            model = model_name or "gpt-3.5-turbo"
            self.client = OpenAIClient(api_key=api_key, model_name=model)
        elif self.provider == LLMProvider.HUGGINGFACE:
            model = model_name or "mistralai/Mistral-7B-Instruct-v0.2"
            self.client = HuggingFaceClient(api_key=api_key, model_name=model)

    def _load_csv(self) -> Optional[pd.DataFrame]:
        """Load combined metadata CSV from processed folder."""
        paths = [
            "data/processed/metadata_all.csv",
            "../data/processed/metadata_all.csv"
        ]
        
        for path in paths:
            if os.path.exists(path):
                return pd.read_csv(path, dtype={'id': str})
        
        print("[WARNING] metadata_all.csv not found! Context lookups will fail.")
        return None

    def generate_response(self, query: str, context_documents: List[Dict[str, Any]], 
                         config: Optional[GenerationConfig] = None) -> GeneratedResponse:
        """Generate response using configured provider."""
        if not config:
            config = GenerationConfig()
            
        start_time = time.time()
        context = self._build_context(context_documents)
        prompt = self._create_rag_prompt(query, context)
        
        try:
            if self.provider == LLMProvider.GEMINI:
                response = self.client.generate_content(prompt, config=config.to_gemini_config())
                text_content = response.text
            elif self.provider == LLMProvider.OPENAI:
                response = self.client.generate_content(prompt, config=config.to_openai_config())
                text_content = response.choices[0].message.content
            elif self.provider == LLMProvider.HUGGINGFACE:
                response = self.client.generate_content(prompt, config=config.to_hf_config())
                text_content = response.text
            
            return GeneratedResponse(
                content=text_content,
                confidence=0.8,
                generation_time=time.time() - start_time,
                token_count=len(text_content.split())
            )
        except Exception as e:
            return GeneratedResponse(
                content=f"Error: {str(e)}",
                confidence=0.0,
                generation_time=time.time() - start_time,
                token_count=0
            )

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            arxiv_id = doc.get('arxiv_id', 'Unknown')
            embedding_index = doc.get('embedding_index', -1)
            
            if self.metadata_df is not None:
                if embedding_index >= 0:
                    try:
                        text = self.metadata_df['prepared_text'].iloc[embedding_index]
                        context_parts.append(f"Document {i} (ID: {arxiv_id}):\n{text}\n")
                        continue
                    except (IndexError, KeyError):
                        pass
                
                try:
                    normalized_id = self._normalize_arxiv_id(arxiv_id)
                    matching_rows = self.metadata_df[self.metadata_df['id'] == normalized_id]
                    if not matching_rows.empty:
                        text = matching_rows.iloc[0]['prepared_text']
                        context_parts.append(f"Document {i} (ID: {arxiv_id}):\n{text}\n")
                        continue
                except Exception:
                    pass
            
            context_parts.append(f"Document {i} (ID: {arxiv_id}):\nNo text available\n")
        
        return "\n".join(context_parts)

    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt."""
        return f"""You are an AI research assistant specializing in ArXiv academic papers.

INSTRUCTIONS:
1. Answer ONLY using information from the provided documents
2. If insufficient information, state: "Based on the provided documents, I cannot fully answer this question."
3. Be precise and academic in tone
4. Synthesize information from multiple papers when relevant
5. Provide technical definitions and context from the papers

RESEARCH DOCUMENTS:
{context}

USER QUESTION: {query}

RESPONSE:"""

    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """Normalize ArXiv ID."""
        try:
            if '.' in arxiv_id:
                parts = arxiv_id.split('.')
                yymm = parts[0].zfill(4)
                return f"{yymm}.{parts[1]}"
            return arxiv_id
        except:
            return arxiv_id
