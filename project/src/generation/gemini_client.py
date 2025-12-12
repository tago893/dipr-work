"""Gemini API Client."""

import os
import google.generativeai as genai
from typing import Optional, Any
from dataclasses import dataclass

@dataclass
class GeminiConfig:
    """Configuration for Gemini generation."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40


class GeminiClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """Initialize Gemini client."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # print(f"✅ Initialized Gemini Client with model: {model_name}")
    
    def generate_content(self, prompt: str, config: Optional[GeminiConfig] = None) -> Any:
        """Generate content using Gemini model."""
        if not config:
            config = GeminiConfig()
            
        generation_config = genai.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            top_k=config.top_k
        )
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response
        except Exception as e:
            # print(f"❌ Gemini API Error: {e}")
            raise e
            
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'provider': 'Google Gemini',
            'api_configured': bool(self.api_key)
        }
