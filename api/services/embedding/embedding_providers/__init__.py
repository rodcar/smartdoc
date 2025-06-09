"""
Embedding providers package for different embedding model backends.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available and properly configured."""
        pass
    
    @abstractmethod
    def generate_image_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Generate an embedding for an image."""
        pass
    
    @abstractmethod
    def generate_text_embedding(self, text: str) -> Dict[str, Any]:
        """Generate an embedding for text."""
        pass
    
    @property
    @abstractmethod
    def supported_modalities(self) -> List[str]:
        """Return list of supported modalities: 'image', 'text', etc."""
        pass


# Import provider implementations
try:
    from .chromadb_provider import ChromaDBProvider
    AVAILABLE_PROVIDERS = [ChromaDBProvider]
except ImportError as e:
    print(f"Warning: Could not import embedding providers: {e}")
    AVAILABLE_PROVIDERS = []

__all__ = ['EmbeddingProvider', 'AVAILABLE_PROVIDERS'] 