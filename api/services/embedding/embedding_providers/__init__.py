"""
Embedding Providers Package

This package contains all embedding provider implementations.
Each provider uses different embedding models/APIs and can be easily swapped.
"""

from .base import EmbeddingProvider
from .chromadb_provider import ChromaDBProvider

# Registry of all available providers
AVAILABLE_PROVIDERS = [
    ChromaDBProvider,
    # Add other providers here as they are implemented
    # OpenAIProvider,
    # CohereProvider,
    # HuggingFaceProvider,
]

__all__ = [
    'EmbeddingProvider',
    'ChromaDBProvider',
    'AVAILABLE_PROVIDERS',
] 