import time
from typing import Dict, Any, Optional, List, Tuple, Callable
from prefect import flow
from prefect.logging import get_run_logger
from prefect.task_runners import ConcurrentTaskRunner

# Import tasks
from ..tasks import (
    scan_directory_for_images,
    initialize_vectordb,
    process_document_pipeline,
    index_batch_documents,
    index_batch_documents_classifier,
    init_text_embedding_provider,
    init_image_embedding_provider
)

# Import flow settings
from ..config.flow_settings import (
    MAX_CONCURRENT_TASKS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_VECTORDB_PATH,
    DEFAULT_COLLECTION_NAME
)


@flow(
    name="document-processing",
    description="Extract text, classify documents, and store in vector database",
    version="1.0.0",
    task_runner=ConcurrentTaskRunner(max_workers=MAX_CONCURRENT_TASKS),
    retries=1,
    retry_delay_seconds=300
)
def document_processing_flow(
    folder_path: str,
    ocr_provider: Optional[str] = None,
    llm_provider: Optional[str] = None,
    vectordb_provider: Optional[str] = None,
    text_embedding_provider: Optional[str] = None,
    image_embedding_provider: Optional[str] = None,
    max_workers: Optional[int] = MAX_CONCURRENT_TASKS,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    vectordb_path: str = DEFAULT_VECTORDB_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> Dict[str, Any]:
    """
    Process documents through OCR, classification, and vector storage pipeline.
    
    Args:
        folder_path: Directory with images to process
        ocr_provider: OCR service to use 
        llm_provider: LLM service to use
        vectordb_provider: Vector DB service to use
        text_embedding_provider: Text embedding service to use
        image_embedding_provider: Image embedding service to use
        max_workers: Max parallel tasks
        embedding_batch_size: Batch size for embeddings
        vectordb_path: Vector DB directory path
        collection_name: Vector DB collection name
        
    Returns:
        Processing results summary of the document processing pipeline results
    """
    logger = get_run_logger()
    start_time = time.time()

    logger.info(f"Configuration: vectordb_path={vectordb_path}, collection={collection_name}")
    total_steps = 4

    try:
        #########################################################
        # Step 1: Initialize all components in parallel (vectordb, directory scan, embedding models)
        #########################################################
        _log_step(1, total_steps, "Initializing vector database, scanning directory, and embedding models (parallel)...", logger)
        
        # Run all initialization tasks in parallel
        vectordb_future = initialize_vectordb.submit(
            db_path=vectordb_path,
            collection_name=collection_name,
            vectordb_provider=vectordb_provider
        )
        image_scan_future = scan_directory_for_images.submit(folder_path)
        text_embedding_init_future = init_text_embedding_provider.submit(text_embedding_provider)
        image_embedding_init_future = init_image_embedding_provider.submit(image_embedding_provider)
        
        # Wait for all tasks to complete
        image_paths = image_scan_future.result()
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

        text_embedding_init_result = text_embedding_init_future.result()
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

        image_embedding_init_result = image_embedding_init_future.result()
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        vectordb_init_result = vectordb_future.result()
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

        if vectordb_init_result['success']:
            logger.info(f"✅ Vector database ready: {vectordb_init_result['action']}")
        else:
            raise Exception(f"Vector database initialization failed: {vectordb_init_result.get('error')}")
        
        # Check text embedding initialization
        if text_embedding_init_result['success']:
            logger.info(f"✅ Text embedding provider ready: {text_embedding_init_result['action']} ({text_embedding_init_result.get('provider_name', 'unknown')})")
        else:
            raise Exception(f"Text embedding provider initialization failed: {text_embedding_init_result.get('error')}")
        
        # Check image embedding initialization
        if image_embedding_init_result['success']:
            logger.info(f"✅ Image embedding provider ready: {image_embedding_init_result['action']} ({image_embedding_init_result.get('provider_name', 'unknown')})")
        else:
            raise Exception(f"Image embedding provider initialization failed: {image_embedding_init_result.get('error')}")
        
        # Check if no valid images found in directory
        if not image_paths:
            raise Exception(f"No valid images found in directory: {folder_path}")
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        #########################################################
        # Step 2: Process documents through chained pipeline (OCR → Classification → Entity Extraction)
        #########################################################
        _log_step(2, total_steps, "Processing documents through parallel streams (OCR→Classification→Entity + Text Embedding + Image Embedding)...", logger)
        
        # Submit all documents to the chained pipeline in parallel
        pipeline_futures = [
            process_document_pipeline.submit(image_path, ocr_provider, text_embedding_provider, image_embedding_provider) 
            for image_path in image_paths
        ]
        
        # Wait for all pipeline results
        pipeline_results = [future.result() for future in pipeline_futures]
        
        # Extract individual results for compatibility with existing code
        extraction_results = []
        classification_results = []
        entity_extraction_results = []
        text_embedding_results = []
        image_embedding_results = []
        
        for result in pipeline_results:
            if result['success']:
                # Extract OCR results
                if result['ocr_result']:
                    extraction_results.append(result['ocr_result'])
                
                # Extract classification results
                if result['classification_result']:
                    classification_results.append(result['classification_result'])
                
                # Extract entity results
                if result['entity_result']:
                    entity_extraction_results.append(result['entity_result'])
                    
                # Extract embedding results
                if result['text_embedding_result']:
                    text_embedding_results.append(result['text_embedding_result'])
                    
                if result['image_embedding_result']:
                    image_embedding_results.append(result['image_embedding_result'])
            else:
                # Handle failed results
                extraction_results.append({
                    'success': False,
                    'image_path': result.get('image_path', 'unknown'),
                    'error': result.get('error', 'Unknown error')
                })
        
        logger.info(f"✅ Completed chained pipeline processing for {len(pipeline_results)} documents")
        logger.info(f"   - OCR results: {len(extraction_results)}")
        logger.info(f"   - Classification results: {len(classification_results)}")  
        logger.info(f"   - Entity extraction results: {len(entity_extraction_results)}")
        logger.info(f"   - Text embedding results: {len(text_embedding_results)}")
        logger.info(f"   - Image embedding results: {len(image_embedding_results)}")
        
        if not extraction_results:
            raise Exception("No successful OCR results found")
        
        #########################################################
        # Step 3: Index to vector database (Includes Embedding Generation)
        #########################################################
        
        _log_step(4, total_steps, "Indexing documents...", logger)
            
        # Create lookup dictionary for entity results
        entity_lookup = {
            result['image_path']: result['entities']
            for result in entity_extraction_results
            if result['success']
        }
        
        # Build documents for indexing in a single pass
        documents_for_indexing = [
            {
                'image_path': ocr_result['image_path'],
                'extracted_text': ocr_result['extracted_text'],
                'document_type': classification_results[i].get('predicted_category', 'unknown'),
                'extracted_entities': entity_lookup.get(ocr_result['image_path'], {})
            }
            for i, ocr_result in enumerate(extraction_results)
            if ocr_result['success']
        ]

        if not documents_for_indexing:
            raise Exception("No documents for indexing found")

        #########################################################
        # 3.1 Index to main collection (smartdoc_documents)
        #########################################################
        logger.info(f"📚 Indexing to main collection: {collection_name}")
        indexing_result = index_batch_documents(
            documents=documents_for_indexing,
            collection_name=collection_name,
            batch_size=min(embedding_batch_size, 25)
        )       
        
        #########################################################
        # 3.2 Index to classifier collections (smartdoc_classifier_images, smartdoc_classifier_text)
        #########################################################
        logger.info(f"🔍 Indexing to classifier collections...")
        classifier_indexing_result = index_batch_documents_classifier(
            documents_data=documents_for_indexing,
            batch_size=min(embedding_batch_size, 25)
        )

        if not classifier_indexing_result['success']:
            raise Exception(f"Classifier collections indexing failed: {classifier_indexing_result.get('error')}")

        #########################################################
        # Generate summary
        #########################################################
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Compile final results
        final_results = {
            'success': True,
            'vectordb_init_result': vectordb_init_result,
            'total_processing_time': round(total_time, 2),
            'extraction_results': extraction_results,  # Individual OCR results for each image
            'classification_results': classification_results,  # Individual classification results
            'entity_extraction_results': entity_extraction_results,  # Individual entity extraction results
            'text_embedding_results': text_embedding_results,  # Individual text embedding results
            'image_embedding_results': image_embedding_results  # Individual image embedding results
        }
        
        # Log final summary
        logger.info("✅ Document processing pipeline completed successfully!")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Images processed: {len(extraction_results)}")
        logger.info(f"Documents classified: {len(classification_results)}")
        logger.info(f"Entities extracted from: {len(entity_extraction_results)} documents")
        logger.info(f"Text embeddings generated: {len(text_embedding_results)}")
        logger.info(f"Image embeddings generated: {len(image_embedding_results)}")
        return final_results
        
    except Exception as e:
        error_msg = f"Document processing pipeline failed: {str(e)}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'error': error_msg
        }
    
def _log_step(step_num: int, total_steps: int, description: str, logger) -> None:
    """Standardized step logging with consistent format."""
    logger.info(f"Step {step_num}/{total_steps}: {description}")