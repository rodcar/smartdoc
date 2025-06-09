# api/services/ocr/ocr_service.py
from typing import Union, BinaryIO, Optional, Dict, Any
from PIL import Image
from .ocr_providers import OCRProvider, AVAILABLE_PROVIDERS


class OCRService:
    """Simple service for OCR text extraction."""
    
    def __init__(self, auto_load_providers: bool = True):
        self.providers = []
        self.default_provider: Optional[OCRProvider] = None
        
        if auto_load_providers:
            self.load_available_providers()
    
    def load_available_providers(self):
        """Load all available providers."""
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
    
    def extract_text_with_confidence(self, image: Union[str, bytes, BinaryIO, Image.Image], 
                                   provider_name: Optional[str] = None) -> Dict[str, Any]:
        """Extract text with confidence scores and metadata."""
        provider = self._get_provider(provider_name)
        return provider.extract_text_with_confidence(image)
    
    def _get_provider(self, provider_name: Optional[str] = None) -> OCRProvider:
        """Get provider by name or return default."""
        if provider_name is None:
            if self.default_provider is None:
                raise ValueError("No providers available")
            return self.default_provider
        
        for provider in self.providers:
            if provider.name == provider_name:
                return provider
        
        raise ValueError(f"Provider {provider_name} not found")


# Lazy singleton instance
_ocr_service = None

def get_ocr_service():
    """Get the OCR service instance (lazy initialization)."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service

# For backwards compatibility, create a property-like access
class LazyOCRService:
    def __getattr__(self, name):
        return getattr(get_ocr_service(), name)

ocr_service = LazyOCRService()