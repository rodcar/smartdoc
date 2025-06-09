from typing import List, Optional, Dict, Any
import os
from .llm_providers import LLMProvider, AVAILABLE_PROVIDERS


class LLMService:
    """
    Service that manages different LLM providers and can use multiple providers
    for various AI tasks like document classification and text generation.
    """
    
    def __init__(self, auto_load_providers: bool = True):
        self.providers: List[LLMProvider] = []
        self.default_provider: Optional[LLMProvider] = None
        
        if auto_load_providers:
            self.load_available_providers()
    
    def load_available_providers(self):
        """Load all available and properly configured providers."""
        for provider_class in AVAILABLE_PROVIDERS:
            try:
                provider = provider_class()
                if provider.is_available():
                    self.providers.append(provider)
                    if self.default_provider is None:
                        self.default_provider = provider
                else:
                    print(f"Provider {provider_class.__name__} is not available")
            except Exception as e:
                print(f"Warning: Could not load {provider_class.__name__}: {e}")
    
    def add_provider(self, provider: LLMProvider):
        """Add a provider instance."""
        if provider.is_available():
            self.providers.append(provider)
            if self.default_provider is None:
                self.default_provider = provider
        else:
            raise ValueError(f"Provider {provider.name} is not available")
    
    def set_default_provider(self, provider_name: str):
        """Set the default provider by name."""
        for provider in self.providers:
            if provider.name == provider_name:
                self.default_provider = provider
                return
        raise ValueError(f"Provider {provider_name} not found")
    
    def get_providers_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded providers."""
        return [
            {
                "name": provider.name,
                "class": provider.__class__.__name__,
                "supported_features": provider.supported_features,
                "is_default": provider == self.default_provider,
                "available": provider.is_available(),
                "model_info": provider.get_model_info()
            }
            for provider in self.providers
        ]
    
    def classify_document(self, 
                         image_base64: str, 
                         extracted_text: str, 
                         categories: List[str],
                         confidence_threshold: float = 0.7,
                         provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify a document using specified or default provider.
        
        Args:
            image_base64: Base64 encoded image
            extracted_text: Text extracted from the image via OCR
            categories: List of possible document categories
            confidence_threshold: Minimum confidence threshold for classification
            provider_name: Specific provider to use (optional)
            
        Returns:
            dict: Classification result
        """
        provider = self._get_provider(provider_name)
        
        if "classification" not in provider.supported_features:
            raise ValueError(f"Provider {provider.name} does not support classification")
        
        return provider.classify_document(image_base64, extracted_text, categories, confidence_threshold)
    
    def extract_entities(self,
                        image_base64: str,
                        extracted_text: str,
                        document_type: str,
                        confidence_threshold: float = 0.7,
                        provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract entities from a document using specified or default provider.
        
        Args:
            image_base64: Base64 encoded image
            extracted_text: Text extracted from the image via OCR
            document_type: The predicted document type to guide entity extraction
            confidence_threshold: Minimum confidence threshold for entity extraction
            provider_name: Specific provider to use (optional)
            
        Returns:
            dict: Entity extraction result
        """
        provider = self._get_provider(provider_name)
        
        if "entity_extraction" not in provider.supported_features:
            raise ValueError(f"Provider {provider.name} does not support entity extraction")
        
        return provider.extract_entities(image_base64, extracted_text, document_type, confidence_threshold)
    
    def generate_text(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using specified or default provider.
        
        Args:
            prompt: Text prompt to generate from
            provider_name: Specific provider to use (optional)
            **kwargs: Provider-specific parameters
            
        Returns:
            str: Generated text
        """
        provider = self._get_provider(provider_name)
        
        if "text_generation" not in provider.supported_features:
            raise ValueError(f"Provider {provider.name} does not support text generation")
        
        return provider.generate_text(prompt, **kwargs)
    
    def classify_document_multi_provider(self, 
                                       image_base64: str, 
                                       extracted_text: str, 
                                       categories: List[str],
                                       confidence_threshold: float = 0.7,
                                       provider_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Classify document using multiple providers for comparison.
        
        Returns:
            dict: Results from each provider
        """
        if provider_names is None:
            providers_to_use = [p for p in self.providers if "classification" in p.supported_features]
        else:
            providers_to_use = [self._get_provider(name) for name in provider_names]
        
        results = {}
        for provider in providers_to_use:
            try:
                result = provider.classify_document(image_base64, extracted_text, categories, confidence_threshold)
                results[provider.name] = result
            except Exception as e:
                results[provider.name] = {
                    'error': str(e),
                    'provider': provider.name,
                    'success': False
                }
        
        return results
    
    def _get_provider(self, provider_name: Optional[str] = None) -> LLMProvider:
        """Get provider by name or return default."""
        if provider_name is None:
            if self.default_provider is None:
                raise ValueError("No LLM providers available")
            return self.default_provider
        
        for provider in self.providers:
            if provider.name == provider_name:
                return provider
        
        raise ValueError(f"Provider {provider_name} not found")


# Lazy singleton instance
_llm_service = None

def get_llm_service():
    """Get the LLM service instance (lazy initialization)."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

# For backwards compatibility, create a property-like access
class LazyLLMService:
    def __getattr__(self, name):
        return getattr(get_llm_service(), name)

llm_service = LazyLLMService() 