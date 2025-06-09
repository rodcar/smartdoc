from typing import Dict, Any, Optional
from prefect import task
from prefect.logging import get_run_logger

# Import other tasks
from . import (
    extract_text_from_image,
    classify_document,
    extract_entities,
    generate_text_embedding,
    generate_image_embedding
)


def _process_single_document_pipeline(image_path: str, ocr_provider: Optional[str], text_embedding_provider: Optional[str], image_embedding_provider: Optional[str], logger) -> Dict[str, Any]:
    """
    Process a single document through parallel streams:
    - Stream 1: OCR → Classification → Entity Extraction  
    - Stream 2: OCR → Text Embedding (parallel to Classification/Entity)
    - Stream 3: Image Embedding (parallel to OCR)
    
    Args:
        image_path: Path to the image file
        ocr_provider: OCR provider to use
        text_embedding_provider: Text embedding provider to use
        image_embedding_provider: Image embedding provider to use
        logger: Prefect logger instance
        
    Returns:
        Combined results from all processing streams
    """
    try:
        # Start Image Embedding immediately (parallel to everything)
        image_embedding_future = generate_image_embedding.submit(image_path, image_embedding_provider)
        
        # Start OCR
        ocr_result = extract_text_from_image(image_path, ocr_provider)
        if not ocr_result['success']:
            # Wait for image embedding to complete before returning
            image_embedding_result = image_embedding_future.result()
            return {
                'image_path': image_path,
                'success': False,
                'error': f"OCR failed: {ocr_result.get('error')}",
                'ocr_result': ocr_result,
                'classification_result': None,
                'entity_result': None,
                'text_embedding_result': None,
                'image_embedding_result': image_embedding_result
            }
        
        extracted_text = ocr_result['extracted_text'] if ocr_result['extracted_text'] else ""
        
        # Start parallel streams after OCR completes
        futures = {}
        
        # Stream 1: Classification → Entity Extraction
        classification_future = classify_document.submit(image_path, extracted_text)
        futures['classification'] = classification_future
        
        # Stream 2: Text Embedding (if we have text)
        if extracted_text.strip():
            text_embedding_future = generate_text_embedding.submit(extracted_text, image_path, text_embedding_provider)
            futures['text_embedding'] = text_embedding_future
        
        # Wait for classification to complete, then start entity extraction
        classification_result = classification_future.result()
        if classification_result['success'] and extracted_text.strip():
            predicted_category = classification_result['predicted_category']
            entity_future = extract_entities.submit(image_path, extracted_text, predicted_category)
            futures['entity'] = entity_future
        
        # Collect all results
        results = {
            'image_path': image_path,
            'ocr_result': ocr_result,
            'classification_result': classification_result
        }
        
        # Wait for entity extraction
        if 'entity' in futures:
            results['entity_result'] = futures['entity'].result()
        else:
            results['entity_result'] = None
        
        # Wait for text embedding
        if 'text_embedding' in futures:
            results['text_embedding_result'] = futures['text_embedding'].result()
        else:
            results['text_embedding_result'] = None
        
        # Wait for image embedding
        results['image_embedding_result'] = image_embedding_future.result()
        
        # Check overall success
        results['success'] = (
            ocr_result['success'] and 
            classification_result['success'] and
            results['image_embedding_result'].get('success', False)
        )
        
        return results
        
    except Exception as e:
        # Make sure to wait for any pending futures
        try:
            image_embedding_result = image_embedding_future.result() if 'image_embedding_future' in locals() else None
        except:
            image_embedding_result = None
            
        return {
            'image_path': image_path,
            'success': False,
            'error': f"Pipeline processing failed: {str(e)}",
            'ocr_result': None,
            'classification_result': None,
            'entity_result': None,
            'text_embedding_result': None,
            'image_embedding_result': image_embedding_result
        }

@task
def process_document_pipeline(image_path: str, ocr_provider: Optional[str] = None, text_embedding_provider: Optional[str] = None, image_embedding_provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Prefect task wrapper for processing a single document through the parallel pipeline streams.
    """
    logger = get_run_logger()
    return _process_single_document_pipeline(image_path, ocr_provider, text_embedding_provider, image_embedding_provider, logger)
