from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    All embedding providers must implement this interface.
    """
    
    @abstractmethod
    def generate_image_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Generate an embedding for an image.
        
        Args:
            image: Image to embed (path, numpy array, or PIL Image)
            
        Returns:
            dict: Embedding result with 'embedding', 'metadata', etc.
        """
        pass
    
    @abstractmethod
    def generate_text_embedding(self, text: str, source_path: str = "") -> Dict[str, Any]:
        """
        Generate an embedding for text.
        
        Args:
            text: Text content to embed
            source_path: Optional source path for metadata
            
        Returns:
            dict: Embedding result with 'embedding', 'metadata', etc.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this embedding provider."""
        pass
    
    @property
    @abstractmethod
    def supported_modalities(self) -> List[str]:
        """Return list of supported modalities: 'image', 'text', etc."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is properly configured and available."""
        pass 