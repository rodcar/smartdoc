import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from prefect import task
from prefect.logging import get_run_logger
from django.conf import settings

from ..config.flow_settings import MAX_RETRIES, RETRY_DELAY_SECONDS, ENTITY_EXTRACTION_MAX_RETRIES, ENTITY_EXTRACTION_RETRY_DELAY
from api.services.llm import llm_service

# Use document types from settings
DOCUMENT_CATEGORIES = settings.DOCUMENT_TYPES

@task(name="classify-document", retries=MAX_RETRIES, retry_delay_seconds=RETRY_DELAY_SECONDS)
def classify_document(image_path: str, extracted_text: str, confidence_threshold: float = 0.7, provider_name: Optional[str] = None) -> Dict[str, Any]:
    """Classify document type using LLM service."""
    logger = get_run_logger()
    # Ensure the image path is properly handled (no URL decoding issues)
    image_path = str(Path(image_path).resolve())
    logger.info(f"Classifying document: {image_path}")
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        llm_result = llm_service.classify_document(
            image_base64=image_base64,
            extracted_text=extracted_text,
            categories=DOCUMENT_CATEGORIES,
            confidence_threshold=confidence_threshold,
            provider_name=provider_name
        )
        
        processing_time = time.time() - start_time
        result = {
            'image_path': image_path,
            'document_type': llm_result.get('document_type', 'unknown'),
            'processing_time_seconds': round(processing_time, 2),
            'all_categories': DOCUMENT_CATEGORIES,
            'success': llm_result.get('success', False),
            'error': llm_result.get('error'),
            'provider_used': provider_name or 'default'
        }
        
        if result['success']:
            logger.info(f"Document classified as '{result['document_type']}' in {processing_time:.2f}s using {result['provider_used']} provider")
        else:
            logger.warning(f"Could not classify document: {image_path} - {result['error']}")
        
        return result
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Document classification failed for {image_path}: {str(e)}"
        logger.error(error_msg)
        return {
            'image_path': image_path,
            'document_type': 'error',
            'processing_time_seconds': round(processing_time, 2),
            'all_categories': DOCUMENT_CATEGORIES,
            'success': False,
            'error': error_msg,
            'provider_used': provider_name or 'default'
        }

@task(name="extract-entities", retries=ENTITY_EXTRACTION_MAX_RETRIES, retry_delay_seconds=ENTITY_EXTRACTION_RETRY_DELAY)
def extract_entities(image_path: str, extracted_text: str, document_type: str, confidence_threshold: float = 0.7, provider_name: Optional[str] = None) -> Dict[str, Any]:
    """Extract entities from document using LLM service."""
    logger = get_run_logger()
    # Ensure the image path is properly handled (no URL decoding issues)
    image_path = str(Path(image_path).resolve())
    logger.info(f"Extracting entities from document: {image_path} (type: {document_type})")
    start_time = time.time()
    
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        llm_result = llm_service.extract_entities(
            image_base64=image_base64,
            extracted_text=extracted_text,
            document_type=document_type,
            confidence_threshold=confidence_threshold,
            provider_name=provider_name
        )
        
        processing_time = time.time() - start_time
        result = {
            'image_path': image_path,
            'document_type': document_type,
            'entities': llm_result.get('entities', {}),
            'processing_time_seconds': round(processing_time, 2),
            'success': llm_result.get('success', False),
            'error': llm_result.get('error'),
            'provider_used': provider_name or 'default',
            'extracted_text_length': len(extracted_text) if extracted_text else 0
        }
        
        if result['success']:
            entities_found = len(result['entities'])
            logger.info(f"Extracted {entities_found} entities from {document_type} document in {processing_time:.2f}s using {result['provider_used']} provider")
        else:
            logger.warning(f"Could not extract entities from document: {image_path} - {result['error']}")
        
        return result
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Entity extraction failed for {image_path}: {str(e)}"
        logger.error(error_msg)
        return {
            'image_path': image_path,
            'document_type': document_type,
            'entities': {},
            'processing_time_seconds': round(processing_time, 2),
            'success': False,
            'error': error_msg,
            'provider_used': provider_name or 'default',
            'extracted_text_length': len(extracted_text) if extracted_text else 0
        }