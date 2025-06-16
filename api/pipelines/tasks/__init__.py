# Import OCR tasks
from .ocr_tasks import (
    scan_directory_for_images,
    extract_text_from_image,
)

# Import embedding tasks
from .embedding_tasks import (
    generate_image_embedding,
    generate_text_embedding,
    init_text_embedding_provider,
    init_image_embedding_provider
)

# Import indexing tasks
from .indexing_tasks import (
    index_documents,
    initialize_vectordb,
    index_documents_classifier
)

# Import image processing tasks
from .image_processing_tasks import (
    classify_document,
    extract_entities
) 

from .process_document_tasks import (
    process_document_pipeline
)