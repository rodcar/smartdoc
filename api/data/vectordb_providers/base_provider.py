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
        """Initialize the vector database provider with default name and empty collections."""
        self.name = self.__class__.__name__.replace("Provider", "").lower()
        self.client = None
        self.collections = {}
    
    @abstractmethod
    def exists(self, **kwargs) -> bool:
        """Check if the vector database or collection exists."""
        pass
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the vector database connection and setup."""
        pass
    
    @abstractmethod
    def create_collection(self, 
                         name: str, 
                         modality: str,
                         embedding_function: Any = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection in the vector database with specified parameters."""
        pass
        
    @abstractmethod
    def index(self, **kwargs) -> bool:
        """Index documents or vectors into the collection."""
        pass
    
    @abstractmethod
    def query(self, **kwargs) -> Dict[str, Any]:
        """Query the vector database and return matching results."""
        pass