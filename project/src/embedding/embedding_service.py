"""Simple embedding service - just like the notebook."""

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class SimpleEmbedder:
    """Simple embedder using SentenceTransformers - matches notebook approach."""
    
    def __init__(self, model_name="allenai-specter"):
        """Initialize with Specter model like in notebook."""
        # print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        else:
            pass
    
    def embed_texts(self, texts, batch_size=1000):
        """Embed list of texts in batches - exactly like notebook."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=True)
                embeddings.extend(batch_embeddings)
                # print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                continue
        
        return np.array(embeddings)
    
    def embed_query(self, query):
        """Embed single query."""
        return self.model.encode([query])
    
    def save_embeddings(self, embeddings, filepath):
        """Save embeddings to file."""
        np.save(filepath, embeddings)
        # print(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath):
        """Load embeddings from file."""
        embeddings = np.load(filepath)
        # print(f"Loaded embeddings from {filepath}, shape: {embeddings.shape}")
        return embeddings
