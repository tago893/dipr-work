"""OpenAI API Client."""

import os
from openai import OpenAI
from typing import Optional, Any
from dataclasses import dataclass

@dataclass
class OpenAIConfig:
    """Configuration for OpenAI generation."""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_content(self, prompt: str, config: Optional[OpenAIConfig] = None) -> Any:
        """Generate content using OpenAI model."""
        if not config:
            config = OpenAIConfig()
            
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty
            )
            
            class ResponseWrapper:
                def __init__(self, content):
                    self.text = content
            
            return ResponseWrapper(response.choices[0].message.content)
            
        except Exception as e:
            raise e
            
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'provider': 'OpenAI',
            'api_configured': bool(self.api_key)
        }
