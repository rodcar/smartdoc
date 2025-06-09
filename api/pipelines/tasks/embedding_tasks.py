"""
Embedding-related Prefect tasks for multimodal document processing.
Refactored to use the embedding service layer.
"""
import os
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from prefect import task
from prefect.logging import get_run_logger
import hashlib

# Set environment variables to avoid PyTorch meta tensor issues
os.environ.setdefault('TORCH_FORCE_DISABLE_META', 'true')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Global variable to track if embedding models have been initialized per worker
_worker_models_initialized = False
_worker_initialization_lock = None

# Lazy import to avoid loading services during module import
from ..config.flow_settings import MAX_CONCURRENT_TASKS

# Import the embedding service
from ...services.embedding.embedding_service import EmbeddingService


def _initialize_worker_models():
    """
    Initialize embedding models globally per worker process.
    This ensures models are loaded once per worker and cached for reuse.
    """
    global _worker_models_initialized, _worker_initialization_lock
    
    if _worker_models_initialized:
        return True
    
    # Use threading lock for thread safety if not already initialized
    if _worker_initialization_lock is None:
        import threading
        _worker_initialization_lock = threading.Lock()
    
    with _worker_initialization_lock:
        # Double-check pattern to avoid race conditions
        if _worker_models_initialized:
            return True
        
        try:
            # Initialize the embedding functions globally using the service instance
            embedding_service = _get_embedding_service()
            embedding_service._initialize_functions()
            _worker_models_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize worker models: {str(e)}")
            return False


def _ensure_worker_models_initialized():
    """
    Ensure worker models are initialized before processing.
    Returns True if models are ready, False otherwise.
    """
    global _worker_models_initialized
    
    if not _worker_models_initialized:
        return _initialize_worker_models()
    return True


def _get_embedding_service():
    """Get embedding service with lazy import."""
    from api.services.embedding import embedding_service
    return embedding_service


@task(
    name="initialize-embedding-service",
    description="Initialize the embedding service and check provider availability",
    tags=["embedding", "initialization", "service"]
)
def initialize_embedding_service() -> Dict[str, Any]:
    """
    Initialize the embedding service and check provider availability.
    
    Returns:
        Dictionary containing service status and provider information
    """
    logger = get_run_logger()
    embedding_service = _get_embedding_service()
    
    try:
        # Get provider information
        providers_info = embedding_service.get_providers_info()
        
        if not providers_info:
            logger.error("No embedding providers available")
            return {
                'success': False,
                'error': 'No embedding providers available',
                'providers': []
            }
        
        # Log available providers
        for provider in providers_info:
            if provider['available']:
                logger.info(f"✅ {provider['name']} provider available (supports: {provider['supported_modalities']})")
            else:
                logger.warning(f"❌ {provider['name']} provider not available")
        
        default_provider = next((p for p in providers_info if p['is_default']), None)
        if default_provider:
            logger.info(f"🎯 Default provider: {default_provider['name']}")
        
        return {
            'success': True,
            'providers': providers_info,
            'default_provider': default_provider['name'] if default_provider else None,
            'available_count': len([p for p in providers_info if p['available']])
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'providers': []
        }


@task(
    name="generate-image-embedding",
    description="Generate an embedding for a single image",
    tags=["embedding", "image", "single"]
)
def generate_image_embedding(
    image_path: str,
    provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an embedding for a single image.
    
    Args:
        image_path: Path to the image file
        provider_name: Optional specific provider to use
        
    Returns:
        Dictionary containing embedding result and metadata
    """
    logger = get_run_logger()
    embedding_service = _get_embedding_service()
    
    try:
        # Ensure worker models are initialized
        if not _ensure_worker_models_initialized():
            logger.warning("Worker models not initialized, proceeding anyway")
        
        logger.info(f"Generating image embedding for {os.path.basename(image_path)}")
        result = embedding_service.generate_image_embedding(image_path, provider_name)
        
        if result['success']:
            logger.info(f"✅ Generated image embedding for {os.path.basename(image_path)}")
        else:
            logger.warning(f"❌ Failed to generate image embedding for {os.path.basename(image_path)}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Exception generating image embedding for {image_path}: {str(e)}")
        return {
            'image_path': image_path,
            'success': False,
            'error': str(e),
            'embedding_type': 'image'
        }


@task(
    name="generate-text-embedding",
    description="Generate an embedding for extracted text",
    tags=["embedding", "text", "single"]
)
def generate_text_embedding(
    text: str,
    source_path: str = "",
    provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate an embedding for extracted text.
    
    Args:
        text: Text content to embed
        source_path: Optional source path for metadata
        provider_name: Optional specific provider to use
        
    Returns:
        Dictionary containing embedding result and metadata
    """
    logger = get_run_logger()
    embedding_service = _get_embedding_service()
    
    try:
        # Ensure worker models are initialized
        if not _ensure_worker_models_initialized():
            logger.warning("Worker models not initialized, proceeding anyway")
        
        if not text or not text.strip():
            logger.warning(f"No text available for embedding from {source_path}")
            return {
                'source_path': source_path,
                'success': False,
                'error': 'No text available for embedding',
                'embedding_type': 'text'
            }
        
        logger.info(f"Generating text embedding for {source_path or 'text content'}")
        result = embedding_service.generate_text_embedding(text, source_path, provider_name)
        
        if result['success']:
            logger.info(f"✅ Generated text embedding for {source_path or 'text content'}")
        else:
            logger.warning(f"❌ Failed to generate text embedding: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Exception generating text embedding: {str(e)}")
        return {
            'source_path': source_path,
            'text_content': text,
            'success': False,
            'error': str(e),
            'embedding_type': 'text'
        }


@task(
    name="generate-dual-embeddings",
    description="Generate both image and text embeddings for a document",
    tags=["embedding", "dual", "multimodal"]
)
def generate_dual_embeddings(
    image_path: str,
    extracted_text: str,
    classification_result: Optional[Dict[str, Any]] = None,
    provider_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate both image and text embeddings for a single document.
    
    Args:
        image_path: Path to the image file
        extracted_text: OCR extracted text
        classification_result: Optional classification result for metadata
        provider_name: Optional specific provider to use
        
    Returns:
        Dictionary containing both embedding results
    """
    logger = get_run_logger()
    embedding_service = _get_embedding_service()
    
    try:
        # Ensure worker models are initialized
        if not _ensure_worker_models_initialized():
            logger.warning("Worker models not initialized, proceeding anyway")
        
        logger.info(f"Generating dual embeddings for {os.path.basename(image_path)}")
        result = embedding_service.generate_dual_embeddings(
            image_path, 
            extracted_text, 
            classification_result, 
            provider_name
        )
        
        if result['success']:
            img_status = "✅" if result.get('has_image_embedding') else "❌"
            txt_status = "✅" if result.get('has_text_embedding') else "❌"
            logger.info(f"Generated dual embeddings for {os.path.basename(image_path)}: Image {img_status}, Text {txt_status}")
        else:
            logger.warning(f"❌ Failed to generate dual embeddings for {os.path.basename(image_path)}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Exception generating dual embeddings for {image_path}: {str(e)}")
        return {
            'document_id': 'unknown',
            'image_path': image_path,
            'success': False,
            'error': str(e),
            'has_image_embedding': False,
            'has_text_embedding': False
        }


@task(
    name="generate-batch-embeddings",
    description="Generate embeddings for a batch of documents in parallel",
    tags=["embedding", "batch", "parallel"]
)
def generate_batch_embeddings(
    documents: List[Dict[str, Any]],
    max_workers: Optional[int] = None,
    provider_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a batch of documents in parallel.
    
    Args:
        documents: List of documents with 'image_path', 'extracted_text', and optional 'classification'
        max_workers: Maximum number of parallel workers
        provider_name: Optional specific provider to use
        
    Returns:
        List of embedding results
    """
    logger = get_run_logger()
    embedding_service = _get_embedding_service()
    
    if not documents:
        logger.warning("No documents to process")
        return []
    
    # Use default max workers if not specified
    if max_workers is None:
        max_workers = min(MAX_CONCURRENT_TASKS, 16)  # Increased to 16 workers for better performance
    
    logger.info(f"Generating embeddings for {len(documents)} documents with {max_workers} workers")
    
    try:
        start_time = time.time()
        results = []
        
        # Process documents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all embedding generation tasks
            future_to_doc = {}
            for doc in documents:
                future = executor.submit(
                    embedding_service.generate_dual_embeddings,
                    doc['image_path'],
                    doc.get('extracted_text', ''),
                    doc.get('classification'),
                    provider_name
                )
                future_to_doc[future] = doc
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_doc, timeout=300):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('success'):
                        logger.debug(f"✅ Generated embeddings for {os.path.basename(doc['image_path'])}")
                    else:
                        logger.warning(f"❌ Failed to generate embeddings for {os.path.basename(doc['image_path'])}: {result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"Exception generating embeddings for {doc['image_path']}: {str(e)}")
                    # Create error result
                    error_result = {
                        'document_id': 'unknown',
                        'image_path': doc['image_path'],
                        'success': False,
                        'error': str(e),
                        'has_image_embedding': False,
                        'has_text_embedding': False,
                        'created_at': time.time()
                    }
                    results.append(error_result)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"Batch embedding generation completed: {successful}/{len(results)} successful in {total_time:.2f}s")
        
        return results
        
    except Exception as e:
        logger.error(f"Exception in batch embedding generation: {str(e)}")
        # Return error results for all documents
        return [{
            'document_id': 'unknown',
            'image_path': doc.get('image_path', 'unknown'),
            'success': False,
            'error': f"Batch processing failed: {str(e)}",
            'has_image_embedding': False,
            'has_text_embedding': False
        } for doc in documents]
    
# Backwards compatibility functions (deprecated - use service methods directly)
def create_document_id(file_path: str) -> str:
    """
    Create a unique document ID based on file path.
    
    DEPRECATED: Use embedding_service methods directly.
    """
    return hashlib.md5(file_path.encode()).hexdigest()


@task(
    name="init-text-embedding-provider",
    description="Initialize text embedding provider",
    tags=["embedding", "initialization", "text"]
)
def init_text_embedding_provider(provider_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Task wrapper for initializing text embedding provider.
    
    Args:
        provider_name: Optional specific provider to use
        
    Returns:
        Dictionary containing initialization result
    """
    logger = get_run_logger()
    try:
        # Initialize the text embedding provider
        result = EmbeddingService.init_text_embedding_provider(provider_name)
        
        # Initialize models globally per worker
        if result['success']:
            models_initialized = _initialize_worker_models()
            if models_initialized:
                logger.info(f"✅ Text embedding provider and worker models initialized: {result.get('provider_name', 'unknown')}")
            else:
                logger.warning(f"⚠️ Text embedding provider initialized but worker models failed to load: {result.get('provider_name', 'unknown')}")
                result['worker_models_initialized'] = False
        else:
            logger.error(f"❌ Text embedding provider initialization failed: {result.get('error')}")
        
        return result
    except Exception as e:
        logger.error(f"Exception initializing text embedding provider: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


@task(
    name="init-image-embedding-provider", 
    description="Initialize image embedding provider",
    tags=["embedding", "initialization", "image"]
)
def init_image_embedding_provider(provider_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Task wrapper for initializing image embedding provider.
    
    Args:
        provider_name: Optional specific provider to use
        
    Returns:
        Dictionary containing initialization result
    """
    logger = get_run_logger()
    try:
        result = EmbeddingService.init_image_embedding_provider(provider_name)
        if result['success']:
            logger.info(f"✅ Image embedding provider initialized: {result.get('provider_name', 'unknown')}")
        else:
            logger.error(f"❌ Image embedding provider initialization failed: {result.get('error')}")
        return result
    except Exception as e:
        logger.error(f"Exception initializing image embedding provider: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }