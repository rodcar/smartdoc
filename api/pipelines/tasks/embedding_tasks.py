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