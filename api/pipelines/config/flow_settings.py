"""
Pipeline-specific settings and configuration.
"""
import os
from pathlib import Path

# OCR Settings
OCR_TIMEOUT = int(os.getenv('OCR_TIMEOUT', 300))  # 5 minutes per image
MAX_CONCURRENT_TASKS = int(os.getenv('MAX_CONCURRENT_TASKS', 16))  # Parallel processing limit

# Processing Pipeline Settings
DEFAULT_EMBEDDING_BATCH_SIZE = 25
DEFAULT_VECTORDB_PATH = "./chromadb"
DEFAULT_COLLECTION_NAME = "smartdoc_documents"
MAX_EMBEDDING_WORKERS = 16  # Maximum workers for embedding generation

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg'}

# Output settings
OUTPUT_FORMAT = 'txt'  # Can be 'txt', 'json', etc.
SAVE_EXTRACTED_TEXT = True

# Error handling
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 60

# File size limits (in bytes)
MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB per image
MIN_IMAGE_SIZE = 1024  # 1KB minimum

def is_supported_image(file_path: Path) -> bool:
    """Check if file is a supported image format."""
    return file_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS

def validate_image_file(file_path: Path) -> bool:
    """Validate image file size and format."""
    if not file_path.exists():
        return False
    
    if not is_supported_image(file_path):
        return False
    
    file_size = file_path.stat().st_size

    # Check file size
    if file_size < MIN_IMAGE_SIZE or file_size > MAX_IMAGE_SIZE:
        return False
    
    return True 

def get_max_workers(max_workers: int = None) -> int:
    """Get the effective max workers, defaulting to MAX_CONCURRENT_TASKS if not provided."""
    return max_workers if max_workers is not None else MAX_CONCURRENT_TASKS