"""
Base VectorDB provider class for standardizing vector database operations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np


class VectorDBProvider(ABC):
    """
    Abstract base class for vector database providers.
    
    This class defines the standard interface for vector database operations
    including indexing and querying.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__.replace("Provider", "").lower()
        self.client = None
        self.collections = {}
    
    @abstractmethod
    def exists(self, **kwargs) -> bool:
        """
        Check if the database exists.
        
        Args:
            **kwargs: Provider-specific parameters (e.g., db_path for ChromaDB)
            
        Returns:
            bool: True if database exists, False otherwise
        """
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the vector database connection.
        
        Args:
            db_path: Path to the database
            **kwargs: Additional configuration parameters
            
        Returns:
            bool: True if initialization successful
        """
        pass
    
    @abstractmethod
    def create_collection(self, 
                         name: str, 
                         modality: str,
                         embedding_function: Any = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new collection for storing embeddings.
        
        Args:
            name: Collection name
            modality: "image" or "text" to specify the embedding type
            embedding_function: Embedding function for the collection
            metadata: Optional metadata for the collection
            
        Returns:
            bool: True if creation successful
        """
        pass
        
    @abstractmethod
    def index(self, **kwargs) -> bool:
        """
        Add embeddings to a collection.
        
        Args:
            **kwargs: Provider-specific parameters such as:
                - collection_name: Name of the collection
                - texts: List of text documents (for text indexing)
                - images: List of image arrays (for image indexing)
                - metadatas: List of metadata dictionaries
                - ids: List of unique identifiers
                - modality: "text", "image", or "multimodal"
                
        Returns:
            bool: True if indexing successful
        """
        pass
    
    @abstractmethod
    def query(self, **kwargs) -> Dict[str, Any]:
        """
        Query a collection.
        
        Args:
            **kwargs: Provider-specific parameters such as:
                - collection_name: Name of the collection
                - query_text: Query text (for text queries)
                - query_image: Query image array (for image queries)
                - n_results: Number of results to return
                - include: What to include in results
                - modality: "text", "image", or "multimodal"
                
        Returns:
            Dictionary containing query results
        """
        pass