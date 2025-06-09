# api/services/ocr_providers/__init__.py
"""
OCR Providers Package

This package contains all OCR provider implementations.
Each provider uses a different OCR service/engine and can be easily swapped.
"""

from .base import OCRProvider
from .tesseract_provider import TesseractProvider
#from .google_vision_provider import GoogleVisionProvider
#from .aws_textract_provider import AWSTextractProvider

# Registry of all available providers
AVAILABLE_PROVIDERS = [
    TesseractProvider,
   #GoogleVisionProvider,
   #AWSTextractProvider,
]

__all__ = [
    'OCRProvider',
    'TesseractProvider',
    #'GoogleVisionProvider', 
    #'AWSTextractProvider',
    'AVAILABLE_PROVIDERS',
]