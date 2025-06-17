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

@task
def process_document_pipeline(image_path: str, ocr_provider: Optional[str] = None, text_embedding_provider: Optional[str] = None, image_embedding_provider: Optional[str] = None) -> Dict[str, Any]:
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
        
    Returns:
        Combined results from all processing streams
    """
    logger = get_run_logger()
    logger.info(f"🔄 Starting pipeline for: {image_path}")
    try:
        # Start Image Embedding immediately (parallel to everything)
        logger.info(f"📸 Starting image embedding for: {image_path}")
        image_embedding_future = generate_image_embedding.submit(image_path, image_embedding_provider)
        
        # Start OCR
        logger.info(f"📄 Starting OCR for: {image_path}")
        ocr_result = extract_text_from_image(image_path, ocr_provider)
        logger.info(f"📄 OCR completed for: {image_path}, success: {ocr_result.get('success')}")
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
        logger.info(f"🔍 Starting classification for: {image_path}")
        classification_future = classify_document.submit(image_path, extracted_text)
        futures['classification'] = classification_future
        
        # Stream 2: Text Embedding (if we have text)
        if extracted_text.strip():
            text_embedding_future = generate_text_embedding.submit(extracted_text, image_path, text_embedding_provider)
            futures['text_embedding'] = text_embedding_future
        
        # Wait for classification to complete, then start entity extraction
        logger.info(f"⏳ Waiting for classification result for: {image_path}")
        classification_result = classification_future.result()
        logger.info(f"🔍 Classification completed for: {image_path}, success: {classification_result.get('success')}, category: {classification_result.get('document_type')}")
        if classification_result['success'] and extracted_text.strip():
            document_type = classification_result['document_type']
            logger.info(f"🏷️ Starting entity extraction for: {image_path}")
            entity_future = extract_entities.submit(image_path, extracted_text, document_type)
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