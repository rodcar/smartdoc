# api/services/ocr_providers/base.py
from abc import ABC, abstractmethod
from typing import Union, BinaryIO, Dict, Any
import base64
import io
from PIL import Image


class OCRProvider(ABC):
    """
    Abstract base class for OCR providers.
    All OCR providers must implement this interface.
    """
    
    @abstractmethod
    def extract_text(self, image: Union[str, bytes, BinaryIO, Image.Image]) -> str:
        """
        Extract text from an image.
        
        Args:
            image: Can be a file path (str), raw bytes, PIL Image, or file-like object
            
        Returns:
            str: Extracted text content
            
        Raises:
            ValueError: If image format is not supported
            Exception: If OCR fails
        """
        pass
    
    @abstractmethod
    def extract_text_with_confidence(self, image: Union[str, bytes, BinaryIO, Image.Image]) -> Dict[str, Any]:
        """
        Extract text with confidence scores and additional metadata.
        
        Args:
            image: Image to process
            
        Returns:
            dict: Contains 'text', 'confidence', and other provider-specific data
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for this OCR provider."""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> list:
        """Return a list of supported image formats."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is properly configured and available."""
        pass
    
    def _prepare_image(self, image: Union[str, bytes, BinaryIO, Image.Image]) -> Image.Image:
        """
        Helper method to convert various image inputs to PIL Image.
        """
        if isinstance(image, str):
            # File path
            return Image.open(image)
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        elif hasattr(image, 'read'):
            # File-like object
            return Image.open(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise ValueError("Unsupported image type")
    
    def _image_to_bytes(self, image: Image.Image, format: str = 'PNG') -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def _image_to_base64(self, image: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string."""
        image_bytes = self._image_to_bytes(image, format)
        return base64.b64encode(image_bytes).decode('utf-8')