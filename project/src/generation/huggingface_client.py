"""HuggingFace Inference API Client."""

import os
import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace generation."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9


class HuggingFaceClient:
    """Client for HuggingFace Inference API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize HuggingFace client."""
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required")
        
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def generate_content(self, prompt: str, config: Optional[HuggingFaceConfig] = None):
        """Generate content using HuggingFace model."""
        if not config:
            config = HuggingFaceConfig()
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": config.temperature,
                "max_new_tokens": config.max_tokens,
                "top_p": config.top_p,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # HuggingFace returns a list with generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                return type('Response', (), {'text': generated_text})()
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        except Exception as e:
            raise e
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'provider': 'HuggingFace',
            'api_configured': bool(self.api_key)
        }
