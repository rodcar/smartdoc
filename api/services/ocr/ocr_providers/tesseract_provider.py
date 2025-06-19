# api/services/ocr_providers/tesseract_provider.py
from typing import Union, BinaryIO, Dict, Any
from PIL import Image
import io
from .base import OCRProvider


class TesseractProvider(OCRProvider):
    """
    Local Tesseract OCR provider.
    Fast, free, works offline, but less accurate than cloud providers.
    """
    
    def __init__(self, language: str = 'eng'):
        self.language = language
    
    @property
    def name(self) -> str:
        return "tesseract"
    
    @property
    def supported_formats(self) -> list:
        return ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    
    def is_available(self) -> bool:
        try:
            import pytesseract
            # Try to get version to verify tesseract is installed
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def extract_text(self, image: Union[str, bytes, BinaryIO, Image.Image]) -> str:
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")
        
        pil_image = self._prepare_image(image)
        
        # Match notebook exactly: simple call with language only
        text = pytesseract.image_to_string(pil_image, lang=self.language)
        
        # Clean up the text exactly like the notebook
        text = text.strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_text_with_confidence(self, image: Union[str, bytes, BinaryIO, Image.Image]) -> Dict[str, Any]:
        try:
            import pytesseract
        except ImportError:
            raise ImportError("pytesseract is required. Install with: pip install pytesseract")
        
        pil_image = self._prepare_image(image)
        
        # Get detailed data for confidence scoring
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Extract text and calculate average confidence
        words = []
        confidences = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Filter out low confidence detections
                words.append(data['text'][i])
                confidences.append(int(data['conf'][i]))
        
        # Combine words and clean up exactly like the notebook
        text = ' '.join(words)
        text = text.strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'text': text,
            'confidence': avg_confidence / 100,  # Convert to 0-1 scale
            'word_count': len(words),
            'provider': self.name,
            'raw_data': data
        }