"""
Embedding services package for multimodal document processing.
"""

from .embedding_service import EmbeddingService
from .embedding_providers import EmbeddingProvider, AVAILABLE_PROVIDERS

# Lazy initialization of the default service instance
_embedding_service = None

def get_embedding_service():
    """Get the embedding service instance (lazy initialization)."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

# For backwards compatibility, create a property-like access
class LazyEmbeddingService:
    def __getattr__(self, name):
        return getattr(get_embedding_service(), name)

embedding_service = LazyEmbeddingService()

__all__ = [
    'EmbeddingService',
    'embedding_service',
    'get_embedding_service',
    'EmbeddingProvider', 
    'AVAILABLE_PROVIDERS'
] 