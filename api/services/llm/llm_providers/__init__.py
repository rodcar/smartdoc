"""
LLM Providers Package

This package contains all LLM provider implementations.
Each provider uses a different LLM service/API and can be easily swapped.
"""

from .base import LLMProvider
from .openai_provider import OpenAIProvider

# Registry of all available providers
AVAILABLE_PROVIDERS = [
    OpenAIProvider,
    # Add other providers here as they are implemented
    # AnthropicProvider,
    # GoogleProvider,
    # AzureOpenAIProvider,
]

__all__ = [
    'LLMProvider',
    'OpenAIProvider',
    'AVAILABLE_PROVIDERS',
] 