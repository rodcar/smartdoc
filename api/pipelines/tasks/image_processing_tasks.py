"""
Image processing-related Prefect tasks for document analysis and classification.
"""
import os
import time
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from prefect import task
from prefect.logging import get_run_logger

from ..config.flow_settings import (
    MAX_RETRIES,
    RETRY_DELAY_SECONDS
)
from api.services.llm import llm_service


@task(
    name="classify-document",
    description="Classify document type using LLM service",
    retries=MAX_RETRIES,
    retry_delay_seconds=RETRY_DELAY_SECONDS,
    tags=["classification", "ai", "image-processing"]
)
def classify_document(
    image_path: str, 
    extracted_text: str, 
    confidence_threshold: float = 0.7,
    provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Classify a document into predefined categories using the LLM service.
    
    Args:
        image_path: Path to the image file
        extracted_text: Text extracted from the image via OCR
        confidence_threshold: Minimum confidence threshold for classification
        provider_name: Optional LLM provider to use (defaults to configured default)
        
    Returns:
        Dictionary containing classification results
    """
    logger = get_run_logger()
    logger.info(f"Classifying document: {image_path}")
    
    # Document categories
    DOCUMENT_CATEGORIES = [
        "advertisement", "budget", "email", "file_folder", "form", 
        "handwritten", "invoice", "letter", "memo", "news_article", 
        "presentation", "questionnaire", "resume", "scientific_publication", 
        "scientific_report", "specification"
    ]
    
    start_time = time.time()
    
    try:
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Use LLM service for classification
        llm_result = llm_service.classify_document(
            image_base64=image_base64,
            extracted_text=extracted_text,
            categories=DOCUMENT_CATEGORIES,
            confidence_threshold=confidence_threshold,
            provider_name=provider_name
        )
        
        processing_time = time.time() - start_time
        
        # Construct the response in the expected format
        result = {
            'image_path': image_path,
            'predicted_category': llm_result.get('predicted_category', 'unknown'),
            'processing_time_seconds': round(processing_time, 2),
            'all_categories': DOCUMENT_CATEGORIES,
            'success': llm_result.get('success', False),
            'error': llm_result.get('error'),
            'provider_used': provider_name or 'default'
        }
        
        if result['success']:
            logger.info(
                f"Document classified as '{result['predicted_category']}' "
                f"in {processing_time:.2f}s using {result['provider_used']} provider"
            )
        else:
            logger.warning(f"Could not classify document: {image_path} - {result['error']}")
        
        return result
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Document classification failed for {image_path}: {str(e)}"
        logger.error(error_msg)
        
        return {
            'image_path': image_path,
            'predicted_category': 'error',
            'processing_time_seconds': round(processing_time, 2),
            'all_categories': DOCUMENT_CATEGORIES,
            'success': False,
            'error': error_msg,
            'provider_used': provider_name or 'default'
        }


@task(
    name="extract-entities",
    description="Extract entities from document using LLM service",
    retries=MAX_RETRIES,
    retry_delay_seconds=RETRY_DELAY_SECONDS,
    tags=["entity-extraction", "ai", "image-processing"]
)
def extract_entities(
    image_path: str,
    extracted_text: str,
    document_type: str,
    confidence_threshold: float = 0.7,
    provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract relevant entities from a document based on its type using the LLM service.
    
    Args:
        image_path: Path to the image file
        extracted_text: Text extracted from the image via OCR
        document_type: The predicted document type to guide entity extraction
        confidence_threshold: Minimum confidence threshold for entity extraction
        provider_name: Optional LLM provider to use (defaults to configured default)
        
    Returns:
        Dictionary containing entity extraction results
    """
    logger = get_run_logger()
    logger.info(f"Extracting entities from document: {image_path} (type: {document_type})")
    
    start_time = time.time()
    
    try:
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Use LLM service for entity extraction
        llm_result = llm_service.extract_entities(
            image_base64=image_base64,
            extracted_text=extracted_text,
            document_type=document_type,
            confidence_threshold=confidence_threshold,
            provider_name=provider_name
        )
        
        processing_time = time.time() - start_time
        
        # Construct the response in the expected format
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
            logger.info(
                f"Extracted {entities_found} entities from {document_type} document "
                f"in {processing_time:.2f}s using {result['provider_used']} provider"
            )
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