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
        
        # Initialize embedding service (auto_load enabled for image querying)
        self.embedding_service = EmbeddingService(auto_load_providers=True)
    
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
                
                # Query text collection if we have extracted text
                text_result = {"classification": None, "confidence": 0.0}
                if extracted_text:
                    text_result = self._query_similar_text_documents(extracted_text)
                
                # Query image collection
                print("🔍 Debug - About to call image search...")
                image_result = self._query_similar_image_documents(temp_file_path)
                print(f"🔍 Debug - Image search returned: {image_result}")
                
                # Step 3: Determine document type using dual confidence approach
                print("📊 Analyzing document type with dual confidence...")
                predicted_document_type = self._determine_final_classification(image_result, text_result)
                
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
    
    def _query_similar_text_documents(self, text: str, n_results: int = 7) -> Dict[str, Any]:
        """Query similar documents from text embedding collection."""
        try:
            # Use the vectordb provider's query method
            if not self.vectordb_service.default_provider:
                print("⚠️ No vector database provider available for text search")
                return {"classification": None, "confidence": 0.0}
            
            result = self.vectordb_service.default_provider.query(
                collection_name="smartdoc_classifier_text",
                query_text=text,
                n_results=n_results,
                modality="text",
                include=["metadatas", "distances"]
            )
            
            if result.get('success') and result.get('results'):
                print(f"🔍 Debug - Raw text query result: {result}")
                metadatas = result['results'].get('metadatas', [[]])[0] if result['results'].get('metadatas') else []
                distances = result['results'].get('distances', [[]])[0] if result['results'].get('distances') else []
                
                print(f"🔍 Debug - Text metadatas length: {len(metadatas)}, distances length: {len(distances)}")
                
                if metadatas and distances:
                    classification, confidence = self._analyze_classification_confidence(metadatas, distances)
                    print(f"📝 Text search: {classification} (confidence: {confidence:.3f})")
                    return {"classification": classification, "confidence": confidence}
                else:
                    print("⚠️ Text search: No metadatas or distances found")
            
            return {"classification": None, "confidence": 0.0}
            
        except Exception as e:
            print(f"⚠️ Error querying similar text documents: {e}")
            return {"classification": None, "confidence": 0.0}
    
    def _query_similar_image_documents(self, image_path: str, n_results: int = 3) -> Dict[str, Any]:
        """Query similar documents from image embedding collection."""
        try:
            print(f"🔍 Debug - Starting image search for: {image_path}")
            # Generate image embedding for the uploaded image
            embedding_result = self.embedding_service.generate_image_embedding(image_path)
            print(f"🔍 Debug - Image embedding result: {embedding_result.get('success')}")
            
            if not embedding_result.get('success'):
                print(f"⚠️ Failed to generate image embedding: {embedding_result.get('error')}")
                return {"classification": None, "confidence": 0.0}
            
            # Extract the embedding vector from the result
            image_embedding = embedding_result.get('embedding')
            if image_embedding is None:
                print("⚠️ No embedding vector in embedding result")
                return {"classification": None, "confidence": 0.0}
            
            # Query using the image embedding
            if not self.vectordb_service.default_provider:
                print("⚠️ No vector database provider available for image search")
                return {"classification": None, "confidence": 0.0}
            
            print(f"🔍 Debug - About to query image collection with embedding length: {len(image_embedding)}")
            
            # Check if collection exists
            provider = self.vectordb_service.default_provider
            if hasattr(provider, 'collections'):
                collection_exists = "smartdoc_classifier_images" in provider.collections
                print(f"🔍 Debug - smartdoc_classifier_images exists: {collection_exists}")
                if collection_exists:
                    collection = provider.collections["smartdoc_classifier_images"]
                    count = collection.count() if hasattr(collection, 'count') else "unknown"
                    print(f"🔍 Debug - smartdoc_classifier_images count: {count}")
            
            result = self.vectordb_service.default_provider.query(
                collection_name="smartdoc_classifier_images",
                query_embeddings=[image_embedding],  # Use embedding, not raw image
                n_results=n_results,
                modality="image",
                include=["metadatas", "distances"]
            )
            print(f"🔍 Debug - Image query completed, success: {result.get('success') if result else 'None'}")
            if result and not result.get('success'):
                print(f"🔍 Debug - Image query error: {result.get('error')}")
            
            if result.get('success') and result.get('results'):
                print(f"🔍 Debug - Raw image query result: {result}")
                metadatas = result['results'].get('metadatas', [[]])[0] if result['results'].get('metadatas') else []
                distances = result['results'].get('distances', [[]])[0] if result['results'].get('distances') else []
                
                print(f"🔍 Debug - Image metadatas length: {len(metadatas)}, distances length: {len(distances)}")
                
                if metadatas and distances:
                    classification, confidence = self._analyze_classification_confidence(metadatas, distances)
                    print(f"🖼️ Image search: {classification} (confidence: {confidence:.3f})")
                    return {"classification": classification, "confidence": confidence}
                else:
                    print("⚠️ Image search: No metadatas or distances found")
            
            return {"classification": None, "confidence": 0.0}
            
        except Exception as e:
            print(f"⚠️ Error querying similar image documents: {e}")
            import traceback
            print(f"🔍 Debug - Image search exception traceback: {traceback.format_exc()}")
            return {"classification": None, "confidence": 0.0}
    
    def _analyze_classification_confidence(self, metadatas: List[Dict], distances: List[float]) -> tuple:
        """Analyze classification confidence based on search results using distance-based voting."""
        if not metadatas or not distances:
            print("⚠️ No metadatas or distances provided")
            return None, 0.0
        
        print(f"🔍 Debug - metadata sample: {metadatas[0] if metadatas else 'empty'}")
        print(f"🔍 Debug - distances: {distances[:3] if len(distances) >= 3 else distances}")
        
        # Count document types weighted by similarity
        doc_type_scores = {}
        total_weight = 0
        
        for metadata, distance in zip(metadatas, distances):
            # Try both 'type' and 'document_type' keys
            doc_type = metadata.get('type') or metadata.get('document_type', 'unknown')
            print(f"🔍 Debug - doc_type: {doc_type}, distance: {distance:.4f}")
            
            if doc_type == 'unknown':
                continue
                
            # Convert distance to similarity (closer = higher similarity)
            similarity = 1 - distance
            # Weight by similarity (higher similarity = more influence)
            weight = max(0, similarity)  # Ensure non-negative
            
            if doc_type not in doc_type_scores:
                doc_type_scores[doc_type] = 0
            doc_type_scores[doc_type] += weight
            total_weight += weight
        
        print(f"🔍 Debug - doc_type_scores: {doc_type_scores}, total_weight: {total_weight}")
        
        if total_weight == 0 or not doc_type_scores:
            print("⚠️ Zero total weight or no valid document types found")
            return None, 0.0
        
        # Normalize scores to get confidence percentages
        for doc_type in doc_type_scores:
            doc_type_scores[doc_type] /= total_weight
        
        # Get the top prediction and its confidence
        predicted_class = max(doc_type_scores, key=doc_type_scores.get)
        confidence = doc_type_scores[predicted_class]
        
        return predicted_class, confidence

    def _determine_final_classification(self, image_result: Dict, text_result: Dict) -> str:
        """Determine final classification based on confidence levels from both modalities."""
        image_class = image_result.get("classification")
        image_conf = image_result.get("confidence", 0.0)
        text_class = text_result.get("classification") 
        text_conf = text_result.get("confidence", 0.0)
        
        # If only one modality has results
        if image_class and not text_class:
            print(f"🎯 Final: {image_class} (image-only, confidence: {image_conf:.3f})")
            return image_class
        elif text_class and not image_class:
            print(f"🎯 Final: {text_class} (text-only, confidence: {text_conf:.3f})")
            return text_class
        elif not image_class and not text_class:
            print("🎯 Final: unknown (no results)")
            return "unknown"
        
        # Both modalities have results - compare confidence
        if image_conf > text_conf:
            winning_modality = "image"
            final_class = image_class
            final_confidence = image_conf
        elif text_conf > image_conf:
            winning_modality = "text"
            final_class = text_class
            final_confidence = text_conf
        else:
            # Tie - prefer image (or could use other tie-breaking logic)
            winning_modality = "image"
            final_class = image_class
            final_confidence = image_conf
        
        print(f"🎯 Final: {final_class} ({winning_modality} wins, confidence: {final_confidence:.3f})")
        return final_class
    
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