from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    All LLM providers must implement this interface.
    """
    
    @abstractmethod
    def classify_document(self, 
                         image_base64: str, 
                         extracted_text: str, 
                         categories: List[str],
                         confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Classify a document into predefined categories.
        
        Args:
            image_base64: Base64 encoded image
            extracted_text: Text extracted from the image via OCR
            categories: List of possible document categories
            confidence_threshold: Minimum confidence threshold for classification
            
        Returns:
            dict: Classification result with 'predicted_category', 'confidence', etc.
        """
        pass
    
    @abstractmethod
    def extract_entities(self,
                        image_base64: str,
                        extracted_text: str,
                        document_type: str,
                        confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Extract relevant entities from a document based on its type.
        
        Args:
            image_base64: Base64 encoded image
            extracted_text: Text extracted from the image via OCR
            document_type: The predicted document type to guide entity extraction
            confidence_threshold: Minimum confidence threshold for entity extraction
            
        Returns:
            dict: Entity extraction result with 'entities', 'confidence', etc.
        """
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Text prompt to generate from
            **kwargs: Provider-specific parameters
            
        Returns:
            str: Generated text response
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this LLM provider."""
        pass
    
    @property
    @abstractmethod
    def supported_features(self) -> List[str]:
        """Return a list of supported features (e.g., 'classification', 'vision', 'text_generation', 'entity_extraction')."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is properly configured and available."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model being used."""
        pass 