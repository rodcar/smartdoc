import os
import tempfile
import base64
from typing import Dict, Any, List
from collections import Counter
from django.core.files.uploadedfile import UploadedFile

# Import required services following the modular architecture
from api.services.ocr import ocr_service
from api.services.vectordb.vectordb_service import VectorDBService
from api.services.embedding.embedding_service import EmbeddingService
from api.services.llm import llm_service


class AnalysisService:
    """
    Analysis service for document processing.
    Contains the main analyze_document method following the README architecture.
    """
    
    def __init__(self):
        """Initialize the service with required components."""
        # Initialize vector database service
        self.vectordb_service = VectorDBService(auto_load_providers=True)
        self.vectordb_service.initialize_database(db_path="./chromadb")
        
        # Initialize embedding service (auto_load disabled for faster startup)
        self.embedding_service = EmbeddingService(auto_load_providers=False)
    
    def analyze_document(self, image_file: UploadedFile) -> Dict[str, Any]:
        """
        Analyze a document image and return structured results.
        
        Following the README architecture:
        1. Extract text using OCR service
        2. Query vectordb for similar documents (text and image embeddings)
        3. Count most repeated document type from similar documents
        4. Extract entities using LLM service
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            Dict containing document_type and entities
        """
        try:
            # Save uploaded file to temporary location for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            try:
                # Step 1: Extract text using OCR service
                print("🔍 Extracting text using OCR service...")
                ocr_result = ocr_service.extract_text_with_confidence(temp_file_path)
                extracted_text = ocr_result.get('text', '').strip()
                
                if not extracted_text:
                    print("⚠️ No text extracted from image")
                    # Still proceed with image-only analysis
                
                # Step 2: Query similar documents from vector database
                print("🔎 Searching for similar documents in vector database...")
                similar_doc_types = []
                
                # Query text collection if we have extracted text
                if extracted_text:
                    text_similar_docs = self._query_similar_text_documents(extracted_text)
                    similar_doc_types.extend(text_similar_docs)
                
                # Query image collection
                image_similar_docs = self._query_similar_image_documents(temp_file_path)
                similar_doc_types.extend(image_similar_docs)
                
                # Step 3: Determine most common document type
                print("📊 Analyzing document type patterns...")
                predicted_document_type = self._get_most_common_document_type(similar_doc_types)
                
                # Step 4: Extract entities using LLM service
                print("🤖 Extracting entities using LLM service...")
                entities = self._extract_entities_with_llm(temp_file_path, extracted_text, predicted_document_type)
                
                # Return results in required format
                return {
                    "success": True,
                    "document_type": predicted_document_type,
                    "entities": entities
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def _query_similar_text_documents(self, text: str, n_results: int = 10) -> List[str]:
        """Query similar documents from text embedding collection."""
        try:
            # Use the vectordb provider's query method
            if not self.vectordb_service.default_provider:
                print("⚠️ No vector database provider available for text search")
                return []
            
            result = self.vectordb_service.default_provider.query(
                collection_name="smartdoc_classifier_text",
                query_text=text,
                n_results=n_results,
                modality="text",
                include=["metadatas", "distances"]
            )
            
            if result.get('success') and result.get('results'):
                metadatas = result['results'].get('metadatas', [[]])[0]
                document_types = [meta.get('type', 'unknown') for meta in metadatas if meta]
                print(f"📝 Found {len(document_types)} similar text documents")
                return document_types
            
            return []
            
        except Exception as e:
            print(f"⚠️ Error querying similar text documents: {e}")
            return []
    
    def _query_similar_image_documents(self, image_path: str, n_results: int = 10) -> List[str]:
        """Query similar documents from image embedding collection."""
        try:
            # Generate image embedding for the uploaded image
            embedding_result = self.embedding_service.generate_image_embedding(image_path)
            
            if not embedding_result.get('success'):
                print(f"⚠️ Failed to generate image embedding: {embedding_result.get('error')}")
                return []
            
            # Load image as array for query
            from PIL import Image
            import numpy as np
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_array = np.array(img)
            
            # Query using the image array
            if not self.vectordb_service.default_provider:
                print("⚠️ No vector database provider available for image search")
                return []
            
            result = self.vectordb_service.default_provider.query(
                collection_name="smartdoc_classifier_images",
                query_image=image_array,
                n_results=n_results,
                modality="image",
                include=["metadatas", "distances"]
            )
            
            if result.get('success') and result.get('results'):
                metadatas = result['results'].get('metadatas', [[]])[0]
                document_types = [meta.get('type', 'unknown') for meta in metadatas if meta]
                print(f"🖼️ Found {len(document_types)} similar image documents")
                return document_types
            
            return []
            
        except Exception as e:
            print(f"⚠️ Error querying similar image documents: {e}")
            return []
    
    def _get_most_common_document_type(self, document_types: List[str]) -> str:
        """Determine the most common document type from similar documents."""
        if not document_types:
            return "unknown"
        
        # Filter out 'unknown' types and count occurrences
        valid_types = [doc_type for doc_type in document_types if doc_type != 'unknown']
        
        if not valid_types:
            return "unknown"
        
        # Count occurrences and get the most common
        type_counts = Counter(valid_types)
        most_common_type = type_counts.most_common(1)[0][0]
        
        print(f"📈 Document type analysis: {dict(type_counts)} -> Predicted: {most_common_type}")
        return most_common_type
    
    def _extract_entities_with_llm(self, image_path: str, extracted_text: str, document_type: str) -> List[Dict[str, str]]:
        """Extract entities using the LLM service."""
        try:
            # Convert image to base64 for LLM service
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Use LLM service to extract entities
            entity_result = llm_service.extract_entities(
                image_base64=image_base64,
                extracted_text=extracted_text or "",
                document_type=document_type
            )
            
            if entity_result.get('success'):
                # Convert to the required format (list of entities)
                entities = entity_result.get('entities', [])
                print(f"🎯 Successfully extracted {len(entities)} entities")
                return entities
            else:
                print(f"⚠️ Entity extraction failed: {entity_result.get('error')}")
                return []
                
        except Exception as e:
            print(f"⚠️ Error extracting entities with LLM: {e}")
            return []


# Create singleton instance
analysis_service = AnalysisService() 