# api/services/search/search_service.py

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from ..vectordb.vectordb_service import VectorDBService
except ImportError:
    from vectordb.vectordb_service import VectorDBService


class SearchService:
    """
    Search service implementing dual confidence-based document classification.
    Adapts the notebook's search_with_dual_confidence logic for production use.
    """
    
    def __init__(self, 
                 vectordb_service: Optional[VectorDBService] = None,
                 db_path: str = "chromadb"):
        """
        Initialize search service.
        
        Args:
            vectordb_service: VectorDB service instance (auto-created if None)
            db_path: Path to ChromaDB database
        """
        self.vectordb = vectordb_service or VectorDBService()
        
        # Initialize database if not already done
        if not self.vectordb.db_path:
            self.vectordb.initialize_database(db_path)
        
        # Collection names
        self.image_collection_name = "smartdoc_classifier_images"
        self.text_collection_name = "smartdoc_classifier_text"
    
    def analyze_classification_confidence(self, search_results: Dict[str, Any]) -> tuple[Optional[str], float]:
        """
        Analyze classification confidence based on search results.
        Uses distance-based voting with confidence scoring.
        
        Args:
            search_results: Results from ChromaDB query
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Handle the wrapped results structure from vectordb service
        if not search_results or not search_results.get('success'):
            return None, 0.0
            
        # Extract the actual ChromaDB results
        results = search_results.get('results', {})
        if not results or not results.get('metadatas') or not results['metadatas']:
            return None, 0.0
        
        # Count document types weighted by similarity
        doc_type_scores = {}
        total_weight = 0
        
        # ChromaDB returns results as lists, get the first (and usually only) query result
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        for metadata, distance in zip(metadatas, distances):
            doc_type = metadata.get('type')  # Using 'type' metadata field
            if not doc_type:
                continue
                
            # Convert distance to similarity using exponential decay
            # This handles both small distances (< 1) and large distances (> 1) better
            similarity = 1 / (1 + distance)  # Always positive, decreases as distance increases
            weight = similarity  # Use similarity directly as weight
            
            if doc_type not in doc_type_scores:
                doc_type_scores[doc_type] = 0
            doc_type_scores[doc_type] += weight
            total_weight += weight
        
        if total_weight == 0:
            return None, 0.0
        
        # Normalize scores to get confidence percentages
        for doc_type in doc_type_scores:
            doc_type_scores[doc_type] /= total_weight
        
        # Get the top prediction and its confidence
        predicted_class = max(doc_type_scores, key=doc_type_scores.get)
        confidence = doc_type_scores[predicted_class]
        
        return predicted_class, confidence
    
    def determine_final_classification(self, 
                                     image_class: Optional[str], 
                                     image_conf: float,
                                     text_class: Optional[str], 
                                     text_conf: float) -> tuple[Optional[str], float, str]:
        """
        Determine final classification based on confidence levels from both modalities.
        
        Args:
            image_class: Predicted class from image modality
            image_conf: Confidence from image modality
            text_class: Predicted class from text modality
            text_conf: Confidence from text modality
            
        Returns:
            Tuple of (final_class, final_confidence, winning_modality)
        """
        # If only one modality has results
        if image_class and not text_class:
            return image_class, image_conf, "image_only"
        elif text_class and not image_class:
            return text_class, text_conf, "text_only"
        elif not image_class and not text_class:
            return None, 0.0, "no_results"
        
        # Both modalities have results - compare confidence
        if image_conf > text_conf:
            return image_class, image_conf, "image"
        elif text_conf > image_conf:
            return text_class, text_conf, "text"
        else:
            # Tie - prefer image (following notebook logic)
            return image_class, image_conf, "image_tie"
    
    def search_with_dual_confidence(self, 
                                   image_file, 
                                   ocr_text: str = "",
                                   image_top_k: int = 3, 
                                   text_top_k: int = 7) -> Dict[str, Any]:
        """
        Search using dual confidence approach adapted from notebook.
        
        Args:
            image_file: Image file (can be file path, PIL Image, or file-like object)
            ocr_text: Pre-extracted OCR text from the image
            image_top_k: Number of results from image collection
            text_top_k: Number of results from text collection
            
        Returns:
            Dict with classification results and confidence analysis
        """
        try:
            # Handle different input types
            if hasattr(image_file, 'name'):
                # File-like object with name
                image_path = getattr(image_file, 'temporary_file_path', None)
                if not image_path:
                    # Save temporary file if needed
                    import tempfile
                    import os
                    suffix = os.path.splitext(image_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        for chunk in image_file.chunks():
                            tmp.write(chunk)
                        image_path = tmp.name
            else:
                # Assume it's a file path
                image_path = str(image_file)
            
            # Load and process query image
            query_image_array = self.vectordb.load_image_as_array(image_path)
            if query_image_array is None:
                return {
                    'success': False,
                    'error': 'Failed to load query image'
                }
            
            # Use provided OCR text
            query_text = ocr_text.strip() if ocr_text else ""
            
            # Search IMAGE collection
            image_classification, image_confidence = None, 0.0
            try:
                image_results = self.vectordb.default_provider.query_image_collection(
                    self.image_collection_name,
                    query_image_array,
                    image_top_k,
                    include=["metadatas", "distances"]
                )
                image_classification, image_confidence = self.analyze_classification_confidence(image_results)
            except Exception as e:
                print(f"Image search failed: {e}")
            
            # Search TEXT collection (if we have text)
            text_classification, text_confidence = None, 0.0
            if query_text and query_text.strip():
                try:
                    text_results = self.vectordb.default_provider.query_text_collection(
                        self.text_collection_name,
                        query_text,
                        text_top_k,
                        include=["metadatas", "distances"]
                    )
                    text_classification, text_confidence = self.analyze_classification_confidence(text_results)
                except Exception as e:
                    print(f"Text search failed: {e}")
            
            # Confidence-based final classification
            final_classification, final_confidence, winning_modality = self.determine_final_classification(
                image_classification, image_confidence,
                text_classification, text_confidence
            )
            
            return {
                'success': True,
                'query_text': query_text,
                'image_classification': image_classification,
                'image_confidence': round(image_confidence, 3),
                'text_classification': text_classification,
                'text_confidence': round(text_confidence, 3),
                'final_classification': final_classification,
                'final_confidence': round(final_confidence, 3),
                'winning_modality': winning_modality,
                'image_top_k': image_top_k,
                'text_top_k': text_top_k
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def classify_document(self, image_file, ocr_text: str = "") -> Dict[str, Any]:
        """
        Classify document and return simplified response.
        
        Args:
            image_file: Image file to classify
            ocr_text: Pre-extracted OCR text from the image
            
        Returns:
            Dict with document type and confidence
        """
        result = self.search_with_dual_confidence(image_file, ocr_text)
        
        if not result['success']:
            return result
        
        return {
            'success': True,
            'document_type': result['final_classification'],
            'confidence': result['final_confidence'],
            'winning_modality': result['winning_modality']
        }


# Lazy singleton instance
_search_service = None

def get_search_service():
    """Get the search service instance (lazy initialization)."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service

# For backwards compatibility, create a property-like access
class LazySearchService:
    def __getattr__(self, name):
        return getattr(get_search_service(), name)

search_service = LazySearchService() 