from .vectordb_service import VectorDBService

_vectordb_service = None

def get_vectordb_service():
    """Get the vectordb service instance (lazy initialization)."""
    global _vectordb_service
    if _vectordb_service is None:
        _vectordb_service = VectorDBService()
    return _vectordb_service

__all__ = ["VectorDBService", "get_vectordb_service"]