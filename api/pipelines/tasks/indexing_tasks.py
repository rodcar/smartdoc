"""
Indexing-related Prefect tasks for dual multimodal document indexing.
Implements dual indexing approach with separate image and text collections.
"""
import os
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from pathlib import Path
from prefect import task
from prefect.logging import get_run_logger

from ..config.flow_settings import MAX_CONCURRENT_TASKS
from api.services.vectordb import get_vectordb_service


@task(
    name="initialize-vectordb",
    description="Check if vector database exists and initialize if needed"
)
def initialize_vectordb(
    db_path: str = "./chromadb",
    collection_name: str = "smartdoc_documents",
    vectordb_provider: Optional[str] = None
) -> Dict[str, Any]:
    logger = get_run_logger()
    logger.info(f"🔍 Checking vector database at: {db_path}")
    
    vectordb_service = get_vectordb_service()
    vectordb_service.set_provider(vectordb_provider)
        
    # Check if database exists
    db_exists = vectordb_service.check_database_exists(db_path=db_path)
        
    if not db_exists:
        logger.info(f"🆕 Creating new vector database")
            
        # Create and initialize new database
        creation_result = vectordb_service.create_and_initialize_database(
            db_path=db_path,
            collection_name=collection_name
        )
        
        if not creation_result:
            raise Exception(f"Failed to create vector database: {creation_result.get('error')}")
        
        logger.info(f"✅ Created new vector database at: {db_path}")
        return {'success': True, 'action': 'created_and_initialized'}
    else:
        logger.info(f"✅ Vector database already exists at: {db_path}")
        
        # Initialize existing database connection
        init_success = vectordb_service.initialize_database(db_path=db_path)

        if not init_success:
            error_msg = f"Failed to initialize existing database at: {db_path}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        return {'success': True, 'action': 'initialized_existing'}


@task(
    name="initialize-vectordb-service",
    description="Initialize the vector database service and setup dual collections",
    tags=["indexing", "vectordb", "initialization", "dual"]
)
def initialize_vectordb_service(
    db_path: str = "./chroma_db_dual",
    image_collection_name: str = "smartdoc_classifier_images", 
    text_collection_name: str = "smartdoc_classifier_text",
    vectordb_provider: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initialize the vector database service and setup dual collections.
    
    Args:
        db_path: Path to the database directory
        image_collection_name: Name for the image collection
        text_collection_name: Name for the text collection
        vectordb_provider: Optional specific vectordb provider to use (e.g., "chromadb")
        
    Returns:
        Dictionary containing initialization and setup results
    """
    logger = get_run_logger()
    vectordb_service = get_vectordb_service()
    
    try:
        logger.info(f"🎯 Initializing vector database service with dual indexing")
        logger.info(f"Database path: {db_path}")
        logger.info(f"Image collection: {image_collection_name}")
        logger.info(f"Text collection: {text_collection_name}")
        
        # Set specific provider if requested
        if vectordb_provider:
            logger.info(f"🔧 Attempting to use provider: {vectordb_provider}")
            provider_set = vectordb_service.set_provider(vectordb_provider)
            if not provider_set:
                available_providers = list(vectordb_service.providers.keys())
                error_msg = f"Requested provider '{vectordb_provider}' not available. Available: {available_providers}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'requested_provider': vectordb_provider,
                    'available_providers': available_providers
                }
            logger.info(f"✅ Using provider: {vectordb_provider}")
        else:
            # Use default provider
            vectordb_service._ensure_providers_loaded()
            if vectordb_service.default_provider:
                logger.info(f"🔧 Using default provider: {vectordb_service.default_provider.name}")
        
        # Get provider information
        providers_info = vectordb_service.get_providers_info()
        
        if not providers_info:
            logger.error("No vector database providers available")
            return {
                'success': False,
                'error': 'No vector database providers available',
                'providers': []
            }
        
        # Log available providers
        for provider in providers_info:
            if provider['available']:
                logger.info(f"✅ {provider['name']} provider available")
            else:
                logger.warning(f"❌ {provider['name']} provider not available")
        
        # Initialize database
        init_success = vectordb_service.initialize_database(db_path=db_path)
        if not init_success:
            logger.error("Failed to initialize vector database")
            return {
                'success': False,
                'error': 'Failed to initialize vector database',
                'providers': providers_info
            }
        
        # Setup dual collections
        logger.info("🎨 Setting up dual collections for image and text embeddings")
        collections_result = vectordb_service.setup_dual_collections(
            image_collection_name=image_collection_name,
            text_collection_name=text_collection_name
        )
        
        if not collections_result['success']:
            logger.error(f"Failed to setup collections: {collections_result.get('error')}")
            return {
                'success': False,
                'error': f"Failed to setup collections: {collections_result.get('error')}",
                'providers': providers_info
            }
        
        logger.info(f"✅ Vector database service initialized successfully")
        logger.info(f"📸 Image collection '{image_collection_name}': {collections_result['image_count']} documents")
        logger.info(f"📝 Text collection '{text_collection_name}': {collections_result['text_count']} documents")
        
        return {
            'success': True,
            'db_path': db_path,
            'providers': providers_info,
            'collections': collections_result,
            'provider_name': collections_result.get('provider')
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize vector database service: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'providers': []
        }


@task(
    name="index-single-document",
    description="Index a single document with multimodal embeddings",
    tags=["indexing", "single", "multimodal"]
)
def index_single_document(
    image_path: str,
    extracted_text: str,
    document_type: str,
    extracted_entities: Optional[Dict[str, Any]] = None,
    entity_extraction_confidence: float = 0.0,
    collection_name: str = "smartdoc_documents"
) -> Dict[str, Any]:
    """
    Index a single document with multimodal embeddings.
    
    Args:
        image_path: Path to the image file
        extracted_text: OCR extracted text
        document_type: Document type for metadata
        extracted_entities: Optional extracted entities
        entity_extraction_confidence: Optional entity extraction confidence
        collection_name: Name of the document collection
        
    Returns:
        Dictionary containing indexing results
    """
    logger = get_run_logger()
    vectordb_service = get_vectordb_service()
    vectordb_service = get_vectordb_service()
    
    try:
        logger.info(f"📋 Indexing document: {os.path.basename(image_path)} (type: {document_type})")
        
        result = vectordb_service.index_single_document(
            image_path=image_path,
            extracted_text=extracted_text,
            document_type=document_type,
            extracted_entities=extracted_entities,
            entity_extraction_confidence=entity_extraction_confidence,
            collection_name=collection_name
        )
        
        if result['success']:
            status_msg = f"✅ Indexed {os.path.basename(image_path)}"
            if extracted_text and extracted_text.strip():
                status_msg += f" [📝 Text: {len(extracted_text)} chars]"
            if result.get('has_entities'):
                status_msg += " [🏷️ Entities]"
            if result.get('document_type_saved'):
                status_msg += f" [📋 DocType: {document_type}]"
            logger.info(status_msg)
            
            # Log document type save status
            if result.get('document_type_saved'):
                logger.info(f"📋 Document type '{document_type}' saved to smartdoc_document_types collection")
                if 'document_type_entry_id' in result:
                    logger.debug(f"   Entry ID: {result['document_type_entry_id']}")
            elif 'document_type_save_error' in result:
                logger.warning(f"⚠️  Failed to save document type: {result['document_type_save_error']}")
            
            # Log entity catalog updates if any
            if 'entities_catalog_update' in result:
                update = result['entities_catalog_update']
                if update['new_entities_added'] > 0:
                    logger.info(f"🏷️  Added {update['new_entities_added']} new entity types to catalog")
                    logger.info(f"📊 Total entity types in catalog: {update['total_entities_catalog']}")
        else:
            logger.warning(f"❌ Failed to index {os.path.basename(image_path)}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Exception indexing document {image_path}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'file_path': image_path
        }


@task(
    name="index-batch-documents",
    description="Index a batch of documents with multimodal embeddings",
    tags=["indexing", "batch", "multimodal"]
)
def index_batch_documents(
    documents: List[Dict[str, Any]],
    collection_name: str = "smartdoc_documents",
    batch_size: int = 25
) -> Dict[str, Any]:
    """
    Index a batch of documents with multimodal embeddings.
    
    Args:
        documents: List of document dictionaries with keys:
            - image_path: Path to the image file
            - extracted_text: OCR extracted text
            - document_type: Document type for metadata
            - extracted_entities: Optional extracted entities (dict)
            - entity_extraction_confidence: Optional entity extraction confidence (float)
        collection_name: Name of the document collection
        batch_size: Number of documents to process at once
        
    Returns:
        Dictionary containing batch indexing results
    """
    logger = get_run_logger()
    vectordb_service = get_vectordb_service()
    
    try:
        total_docs = len(documents)
        logger.info(f"🚀 Starting multimodal indexing of {total_docs} documents")
        logger.info(f"📊 Batch size: {batch_size}")
        logger.info(f"📚 Collection: {collection_name}")
        
        # Process in batches for better memory management
        total_indexed = 0
        total_failed = 0
        total_entities_updates = 0
        batch_results = []
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")
            
            # Index this batch
            batch_result = vectordb_service.index_batch_documents(
                documents=batch_docs,
                collection_name=collection_name
            )
            
            batch_results.append(batch_result)
            
            if batch_result['success']:
                batch_indexed = batch_result['indexed_documents']
                batch_failed = batch_result['failed_count']
                batch_entities_updates = batch_result.get('entities_catalog_updates', 0)
                
                total_indexed += batch_indexed
                total_failed += batch_failed
                total_entities_updates += batch_entities_updates
                
                logger.info(f"✅ Batch {batch_num}: {batch_indexed} indexed, {batch_failed} failed")
                
                # Log document types saved
                doc_types_saved = batch_result.get('document_types_saved', 0)
                doc_types_failed = batch_result.get('document_types_failed', 0)
                if doc_types_saved > 0:
                    logger.info(f"📋 Document types saved: {doc_types_saved}")
                if doc_types_failed > 0:
                    logger.warning(f"⚠️  Document types save failed: {doc_types_failed}")
                
                if batch_entities_updates > 0:
                    logger.info(f"🏷️  Added {batch_entities_updates} new entity types to catalog")
            else:
                batch_failed = len(batch_docs)
                total_failed += batch_failed
                logger.error(f"❌ Batch {batch_num} failed: {batch_result.get('error')}")
        
        success_rate = (total_indexed / total_docs * 100) if total_docs > 0 else 0
        
        logger.info(f"🎯 Batch indexing completed:")
        logger.info(f"   ✅ Indexed: {total_indexed}")
        logger.info(f"   ❌ Failed: {total_failed}")
        logger.info(f"   📊 Success rate: {success_rate:.1f}%")
        if total_entities_updates > 0:
            logger.info(f"   🏷️  Total new entity types: {total_entities_updates}")
        
        return {
            'success': total_indexed > 0,
            'total_indexed': total_indexed,
            'indexed_images': total_indexed,  # Since we're using image embeddings for smartdoc_documents
            'indexed_texts': 0,  # No separate text indexing for smartdoc_documents
            'failed_count': total_failed,
            'total_failed': total_failed,
            'total_processed': total_docs,
            'success_rate': round(success_rate, 2),
            'entities_catalog_updates': total_entities_updates,
            'collection_name': collection_name,
            'batch_results': batch_results,
            'provider': vectordb_service.default_provider.name if vectordb_service.default_provider else None
        }
        
    except Exception as e:
        logger.error(f"Exception during batch indexing: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'total_indexed': 0,
            'indexed_images': 0,
            'indexed_texts': 0,
            'failed_count': len(documents),
            'total_failed': len(documents),
            'total_processed': len(documents),
            'collection_name': collection_name
        }


@task(
    name="index-document-to-classifiers",
    description="Index a single document to classifier collections",
    tags=["indexing", "single", "classifier", "dual"]
)
def index_document_to_classifiers(
    image_path: str,
    extracted_text: str,
    document_type: str,
    image_collection_name: str = "smartdoc_classifier_images",
    text_collection_name: str = "smartdoc_classifier_text",
    extracted_entities: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a single document to both classifier collections (image and text embeddings).
    
    Args:
        image_path: Path to the image file
        extracted_text: OCR extracted text
        document_type: Document type for metadata
        image_collection_name: Name of the image collection for classification
        text_collection_name: Name of the text collection for classification
        extracted_entities: Optional extracted entities
        
    Returns:
        Dictionary containing dual indexing results
    """
    logger = get_run_logger()
    vectordb_service = get_vectordb_service()
    
    try:
        logger.info(f"📋 Indexing to classifiers: {os.path.basename(image_path)} (type: {document_type})")
        
        result = vectordb_service.index_document_to_classifier_collections(
            image_path=image_path,
            extracted_text=extracted_text,
            document_type=document_type,
            image_collection_name=image_collection_name,
            text_collection_name=text_collection_name,
            extracted_entities=extracted_entities
        )
        
        if result['success']:
            status_msg = f"✅ Indexed {os.path.basename(image_path)}"
            if result.get('image_indexed'):
                status_msg += " [📸 Image]"
            if result.get('text_indexed'):
                status_msg += " [📝 Text]"
            logger.info(status_msg)
            
            # Log individual indexing results
            if result.get('image_indexed'):
                logger.info(f"📸 Image indexed to '{image_collection_name}' collection")
            elif result.get('image_error'):
                logger.warning(f"⚠️  Image indexing failed: {result['image_error']}")
                
            if result.get('text_indexed'):
                logger.info(f"📝 Text indexed to '{text_collection_name}' collection")
            elif result.get('text_error'):
                logger.warning(f"⚠️  Text indexing failed: {result['text_error']}")
        else:
            logger.warning(f"❌ Failed to index {os.path.basename(image_path)}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Exception indexing document to classifiers {image_path}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'file_path': image_path,
            'image_indexed': False,
            'text_indexed': False
        }


@task(
    name="index-batch-documents-classifier",
    description="Process documents with OCR extraction and dual indexing",
    tags=["indexing", "ocr", "dual", "parallel"]
)
def index_batch_documents_classifier(
    documents_data: List[Dict[str, Any]],
    image_collection_name: str = "smartdoc_classifier_images",
    text_collection_name: str = "smartdoc_classifier_text",
    max_workers: Optional[int] = None,
    batch_size: int = 25
) -> Dict[str, Any]:
    """
    Process documents with OCR extraction and dual indexing in parallel.
    
    This combines the OCR extraction and indexing steps for efficiency.
    Each worker thread gets its own ChromaDB client to prevent context leaks.
    
    Args:
        documents_data: List of document dictionaries with keys:
            - image_path: Path to the image file
            - document_type: Document type for metadata
            - extracted_text: Optional pre-extracted text
        image_collection_name: Name of the image collection
        text_collection_name: Name of the text collection
        max_workers: Number of parallel workers (None = auto-detect)
        batch_size: Number of documents to process in each batch
        
    Returns:
        Dictionary containing processing and indexing results
    """
    logger = get_run_logger()
    
    try:
        # Auto-detect optimal worker count
        if max_workers is None:
            from multiprocessing import cpu_count
            max_workers = min(cpu_count(), 16)  # Allow up to 16 workers with thread-local clients
        
        total_docs = len(documents_data)
        logger.info(f"🚀 Starting sequential OCR + dual indexing of {total_docs} documents")
        logger.info(f"🔧 Configuration: batch_size={batch_size} (sequential processing)")
        logger.info(f"📸 Image collection: {image_collection_name}")
        logger.info(f"📝 Text collection: {text_collection_name}")
        
        def process_single_document(doc_data):
            """Process a single document with OCR + indexing."""
            image_path = doc_data['image_path']
            document_type = doc_data['document_type']
            extracted_text = doc_data.get('extracted_text', '')
            
            try:
                # Use the main vectordb service (no threading, so safe)
                vectordb_service = get_vectordb_service()
                result = vectordb_service.index_document_to_classifier_collections(
                    image_path=image_path,
                    extracted_text=extracted_text,
                    document_type=document_type,
                    image_collection_name=image_collection_name,
                    text_collection_name=text_collection_name,
                    extracted_entities=doc_data.get('extracted_entities')
                )
                
                return result
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'file_path': image_path,
                    'image_indexed': False,
                    'text_indexed': False
                }
        
        # Process documents in parallel batches with thread-local clients
        total_indexed_images = 0
        total_indexed_texts = 0
        total_failed = 0
        processing_results = []
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents_data[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"🔍 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")
            
            # Process batch sequentially to avoid ChromaDB context leaks
            batch_results = []
            for doc_data in batch_docs:
                result = process_single_document(doc_data)
                batch_results.append(result)
            
            # Aggregate results
            batch_image_count = sum(1 for r in batch_results if r.get('image_indexed', False))
            batch_text_count = sum(1 for r in batch_results if r.get('text_indexed', False))
            batch_failed_count = sum(1 for r in batch_results if not r.get('success', False))
            
            total_indexed_images += batch_image_count
            total_indexed_texts += batch_text_count
            total_failed += batch_failed_count
            
            processing_results.extend(batch_results)
            
            logger.info(f"✅ Batch {batch_num} completed:")
            logger.info(f"   📸 {batch_image_count} image embeddings")
            logger.info(f"   📝 {batch_text_count} text embeddings")
            logger.info(f"   ❌ {batch_failed_count} failed")
        
        # Get final collection stats using main service
        # main_vectordb_service = get_vectordb_service()
        # stats = main_vectordb_service.get_collection_stats(
        #     image_collection_name=image_collection_name,
        #     text_collection_name=text_collection_name
        # )
        
        logger.info(f"🎉 Sequential OCR + dual indexing completed!")
        logger.info(f"📈 Total processed: {total_docs}")
        logger.info(f"✅ Image embeddings created: {total_indexed_images}")
        logger.info(f"✅ Text embeddings created: {total_indexed_texts}")
        logger.info(f"❌ Failed: {total_failed}")
        
        #if stats['success']:
            #logger.info(f"📚 Final collection stats:")
            #logger.info(f"   📸 {image_collection_name}: {stats['image_count']} embeddings")
            #logger.info(f"   📝 {text_collection_name}: {stats['text_count']} embeddings")
        
        return {
            'success': True,
            'total_documents': total_docs,
            'indexed_images': total_indexed_images,
            'indexed_texts': total_indexed_texts,
            'failed_count': total_failed,
            'processing_results': processing_results,
            #'collection_stats': stats,
            'image_collection': image_collection_name,
            'text_collection': text_collection_name,
            'max_workers': max_workers,
            'batch_size': batch_size
        }
        
    except Exception as e:
        logger.error(f"Exception during parallel processing + indexing: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'total_documents': len(documents_data),
            'indexed_images': 0,
            'indexed_texts': 0,
            'failed_count': len(documents_data)
        }


@task(
    name="search-with-dual-confidence",
    description="Search using dual confidence approach with image and text queries",
    tags=["search", "dual", "confidence", "multimodal"]
)
def search_with_dual_confidence(
    query_image_path: str,
    query_text: Optional[str] = None,
    image_collection_name: str = "smartdoc_classifier_images",
    text_collection_name: str = "smartdoc_classifier_text",
    n_results: int = 10
) -> Dict[str, Any]:
    """
    Search using dual confidence approach with separate image and text queries.
    
    Args:
        query_image_path: Path to query image
        query_text: Optional query text
        image_collection_name: Name of the image collection
        text_collection_name: Name of the text collection
        n_results: Number of results to return from each collection
        
    Returns:
        Dictionary containing dual search results
    """
    logger = get_run_logger()
    vectordb_service = get_vectordb_service()
    
    try:
        logger.info(f"🔍 Dual confidence search for: {os.path.basename(query_image_path)}")
        if query_text:
            logger.info(f"📝 Query text provided: {len(query_text)} characters")
        
        result = vectordb_service.search_with_dual_confidence(
            query_image_path=query_image_path,
            query_text=query_text,
            image_collection_name=image_collection_name,
            text_collection_name=text_collection_name,
            n_results=n_results
        )
        
        if result['success']:
            logger.info(f"✅ Dual search completed successfully")
            
            # Log search results
            image_results = result.get('image_results')
            text_results = result.get('text_results')
            
            if image_results and image_results.get('success'):
                logger.info(f"📸 Found {len(image_results['results']['metadatas'][0]) if image_results['results']['metadatas'] else 0} image matches")
            
            if text_results and text_results.get('success'):
                logger.info(f"📝 Found {len(text_results['results']['metadatas'][0]) if text_results['results']['metadatas'] else 0} text matches")
        else:
            logger.warning(f"❌ Dual search failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Exception during dual confidence search: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'query_image': query_image_path,
            'query_text': query_text
        }

def create_document_id(file_path: str) -> str:
    """Create a unique document ID based on file path."""
    vectordb_service = get_vectordb_service()
    return vectordb_service.create_document_id(file_path) 