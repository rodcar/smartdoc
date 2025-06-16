"""
Indexing tasks for document processing with separate image and text collections.
"""
import os
from typing import List, Dict, Any, Optional
from prefect import task
from prefect.logging import get_run_logger
from api.services.vectordb import get_vectordb_service


@task(
    name="initialize-vectordb"
)
def initialize_vectordb(
    db_path: str = "./chromadb",
    collection_name: str = "smartdoc_documents",
    vectordb_provider: Optional[str] = None
) -> Dict[str, Any]:
    """Initialize vector database and create if it doesn't exist."""
    logger = get_run_logger()
    logger.info(f"🔍 Checking vector database at: {db_path}")
    
    vectordb_service = get_vectordb_service()
    vectordb_service.set_provider(vectordb_provider)
    
    db_exists = vectordb_service.check_database_exists(db_path=db_path)
    
    if not db_exists:
        logger.info(f"🆕 Creating new vector database")
        creation_result = vectordb_service.create_and_initialize_database(
            db_path=db_path,
            collection_name=collection_name
        )
        
        if not creation_result:
            raise Exception(f"Failed to create vector database: {creation_result.get('error')}")
        
        logger.info(f"✅ Created new vector database at: {db_path}")
        return {'success': True, 'action': 'created_and_initialized'}
    
    logger.info(f"✅ Vector database already exists at: {db_path}")
    init_success = vectordb_service.initialize_database(db_path=db_path)
    
    if not init_success:
        error_msg = f"Failed to initialize existing database at: {db_path}"
        logger.error(error_msg)
        return {'success': False, 'error': error_msg}
    
    return {'success': True, 'action': 'initialized_existing'}


@task(
    name="index-documents",
    description="Index multiple documents"
)
def index_documents(
    documents: List[Dict[str, Any]],
    collection_name: str = "smartdoc_documents",
    batch_size: int = 25
) -> Dict[str, Any]:
    """Index a batch of documents with their metadata and extracted text."""
    logger = get_run_logger()
    vectordb_service = get_vectordb_service()
    
    try:
        total_docs = len(documents)
        logger.info(f"🚀 Starting indexing of {total_docs} documents")
        
        total_indexed = 0
        total_failed = 0
        batch_results = []
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")
            
            batch_result = vectordb_service.index_documents(
                documents=batch_docs,
                collection_name=collection_name
            )
            
            batch_results.append(batch_result)
            
            if batch_result['success']:
                batch_indexed = batch_result['indexed_documents']
                batch_failed = batch_result['failed_count']
                
                total_indexed += batch_indexed
                total_failed += batch_failed
                
                logger.info(f"✅ Batch {batch_num}: {batch_indexed} indexed, {batch_failed} failed")
            else:
                batch_failed = len(batch_docs)
                total_failed += batch_failed
                logger.error(f"❌ Batch {batch_num} failed: {batch_result.get('error')}")
        
        success_rate = (total_indexed / total_docs * 100) if total_docs > 0 else 0
        
        logger.info(f"🎯 Batch indexing completed:")
        logger.info(f"   ✅ Indexed: {total_indexed}")
        logger.info(f"   ❌ Failed: {total_failed}")
        logger.info(f"   📊 Success rate: {success_rate:.1f}%")
        
        return {
            'success': total_indexed > 0,
            'total_indexed': total_indexed,
            'failed_count': total_failed,
            'total_processed': total_docs,
            'success_rate': round(success_rate, 2),
            'collection_name': collection_name,
            'batch_results': batch_results
        }
        
    except Exception as e:
        logger.error(f"Exception during batch indexing: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'total_indexed': 0,
            'failed_count': len(documents),
            'total_processed': len(documents),
            'collection_name': collection_name
        }


@task(
    name="index-documents-classifier",
    description="Index multiple documents to classifier collections (image and text)"
)
def index_documents_classifier(
    documents_data: List[Dict[str, Any]],
    batch_size: int = 25
) -> Dict[str, Any]:
    """Index a batch of documents to classifier collections (image and text)."""
    logger = get_run_logger()
    vectordb_service = get_vectordb_service()
    total_docs = len(documents_data)
    logger.info(f"🚀 Starting classifier indexing of {total_docs} documents")

    total_indexed = 0
    total_failed = 0
    batch_results = []

    for i in range(0, total_docs, batch_size):
        batch_docs = documents_data[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        logger.info(f"📦 Processing classifier batch {batch_num}/{total_batches} ({len(batch_docs)} documents)")
        batch_result = []
        for doc in batch_docs:
            try:
                result = vectordb_service.index_document_to_classifier_collections(
                    image_path=doc.get('image_path'),
                    extracted_text=doc.get('extracted_text', ''),
                    document_type=doc.get('document_type', 'unknown'),
                    extracted_entities=doc.get('extracted_entities', None)
                )
                batch_result.append(result)
                if result.get('success'):
                    total_indexed += 1
                else:
                    total_failed += 1
            except Exception as e:
                logger.error(f"❌ Error indexing document to classifier collections: {e}")
                total_failed += 1
                batch_result.append({'success': False, 'error': str(e), 'image_path': doc.get('image_path')})
        batch_results.append(batch_result)
        logger.info(f"✅ Classifier batch {batch_num}: {len([r for r in batch_result if r.get('success')])} indexed, {len([r for r in batch_result if not r.get('success')])} failed")

    success_rate = (total_indexed / total_docs * 100) if total_docs > 0 else 0
    logger.info(f"🎯 Classifier batch indexing completed:")
    logger.info(f"   ✅ Indexed: {total_indexed}")
    logger.info(f"   ❌ Failed: {total_failed}")
    logger.info(f"   📊 Success rate: {success_rate:.1f}%")

    return {
        'success': total_indexed > 0,
        'total_indexed': total_indexed,
        'failed_count': total_failed,
        'total_processed': total_docs,
        'success_rate': round(success_rate, 2),
        'batch_results': batch_results
    } 