# Import OCR tasks
from .ocr_tasks import (
    scan_directory_for_images,
    extract_text_from_image,
)

# Import embedding tasks
from .embedding_tasks import (
    initialize_embedding_service,
    generate_batch_embeddings,
    generate_dual_embeddings,
    generate_image_embedding,
    generate_text_embedding,
    init_text_embedding_provider,
    init_image_embedding_provider
)

# Import indexing tasks
from .indexing_tasks import (
    initialize_vectordb_service,
    index_single_document,
    index_batch_documents,
    index_batch_documents_classifier,
    search_with_dual_confidence,
    create_document_id,
    initialize_vectordb
)

# Import image processing tasks
from .image_processing_tasks import (
    classify_document,
    extract_entities
) 

from .process_document_tasks import (
    process_document_pipeline
)