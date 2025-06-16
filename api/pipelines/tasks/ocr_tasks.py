"""
OCR-related Prefect tasks for image text extraction.
"""
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from prefect import task
from prefect.logging import get_run_logger

from ..config.flow_settings import (
    OCR_TIMEOUT, 
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    is_supported_image,
    validate_image_file
)

# Import OCR service for provider-agnostic processing
try:
    from api.services.ocr import ocr_service
except ImportError:
    ocr_service = None


@task(
    name="scan-directory",
    description="Scan directory recursively for image files",
)
def scan_directory_for_images(folder_path: str) -> List[str]:
    """
    Scan a directory recursively and return all supported image file paths.
    
    Args:
        folder_path: Root directory path to scan
        
    Returns:
        List of valid image file paths
    """
    logger = get_run_logger()
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Directory does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    logger.info(f"Scanning directory: {folder_path}")
    
    image_files = []
    total_scanned = 0
    
    # Recursively find all image files
    for file_path in folder.rglob("*"):
        if file_path.is_file():
            total_scanned += 1
            if validate_image_file(file_path):
                image_files.append(str(file_path))
                logger.debug(f"Found valid image: {file_path}")
    
    logger.info(f"Scanned {total_scanned} files, found {len(image_files)} valid images")
    return image_files


@task(
    name="extract-text-from-image",
    description="Extract text from a single image using OCR service",
    retries=MAX_RETRIES,
    retry_delay_seconds=RETRY_DELAY_SECONDS,
)
def extract_text_from_image(image_path: str, provider_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract text from a single image using the OCR service.
    
    Args:
        image_path: Path to the image file
        provider_name: Optional specific OCR provider to use
        
    Returns:
        Dictionary containing extraction results
    """
    logger = get_run_logger()
    image_file = Path(image_path)
    
    logger.info(f"Processing image: {image_path}")
    start_time = time.time()
    
    try:
        # Check if OCR service is available
        if ocr_service is None:
            raise ImportError("OCR service is not available")
        
        # Use provider from environment variable if not specified
        if provider_name is None:
            provider_name = os.environ.get('OCR_PROVIDER')
        
        logger.debug(f"Using OCR provider: {provider_name or 'default'}")
        
        # Extract text using OCR service
        result = ocr_service.extract_text_with_confidence(
            image=str(image_file),
            provider_name=provider_name
        )
        
        processing_time = time.time() - start_time
        
        # Prepare response in consistent format
        extraction_result = {
            'image_path': image_path,
            'extracted_text': result['text'],
            'text_length': len(result['text']),
            'processing_time_seconds': round(processing_time, 2),
            'confidence': result.get('confidence', 0.0),
            'provider': result.get('provider', 'Unknown'),
            'word_count': result.get('word_count', 0),
            'success': True,
            'error': None
        }
        
        logger.info(
            f"Successfully extracted {len(result['text'])} characters "
            f"in {processing_time:.2f}s from {image_file.name} "
            f"using {result.get('provider', 'Unknown')}"
        )
        
        return extraction_result
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"OCR failed for {image_path}: {str(e)}"
        logger.error(error_msg)
        
        return {
            'image_path': image_path,
            'extracted_text': '',
            'text_length': 0,
            'processing_time_seconds': round(processing_time, 2),
            'confidence': 0.0,
            'provider': 'Error',
            'word_count': 0,
            'success': False,
            'error': error_msg
        }
 