"""
ChromaDB provider for vector database operations.
Implements dual indexing with separate image and text collections.
"""
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from .base_provider import VectorDBProvider


class ChromaDBProvider(VectorDBProvider):
    """
    ChromaDB implementation of the VectorDBProvider.
    
    Supports indexing with separate collections for:
    - Image embeddings (using OpenCLIP)
    - Text embeddings (using SentenceTransformer)
    """
    
    def __init__(self):
        super().__init__()
        self.name = "chromadb"
        
    def exists(self, **kwargs) -> bool:
        """Check if ChromaDB database exists at the given path."""
        try:
            from pathlib import Path
            db_path = kwargs.get('db_path', '')
            path = Path(db_path)
            return path.exists() and path.is_dir() and any(path.glob("*.sqlite*"))
        except Exception:
            return False
    
    def initialize(self, **kwargs) -> bool:
        """Initialize ChromaDB client with persistent storage."""
        try:
            import chromadb
            db_path = kwargs.get('db_path', '')
            os.makedirs(db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path, **{k: v for k, v in kwargs.items() if k != 'db_path'})

            # Create collections without embedding functions
            # We'll compute embeddings manually during indexing for more control
            self.create_collection("smartdoc_documents", "text")
            self.create_collection("smartdoc_document_types", "text")  
            self.create_collection("smartdoc_classifier_images", "image")
            self.create_collection("smartdoc_classifier_text", "text")

            return True
        except Exception as e:
            print(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def create_collection(self, name: str, modality: str, embedding_function: Any = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection or get existing one."""
        try:    
            collection_metadata = metadata or {}
            collection_metadata.update({
                "modality": modality,
                "created_at": datetime.now().isoformat()
            })
            
            try:
                # Try to create new collection without embedding function
                collection = self.client.create_collection(name=name, metadata=collection_metadata)
            except Exception:
                # If creation fails (likely because it exists), try to get existing collection
                try:
                    collection = self.client.get_collection(name=name)
                except Exception as get_e:
                    print(f"Failed to create or get collection {name}: {get_e}")
                    return False
            
            self.collections[name] = collection
            return True
        except Exception as e:
            print(f"Failed to create collection {name}: {e}")
            return False

    def index(self, **kwargs) -> bool:
        """Add embeddings to a collection."""
        try:
            collection_name = kwargs.get('collection_name')
            modality = kwargs.get('modality', 'text')
            texts = kwargs.get('texts', [])
            images = kwargs.get('images', [])
            metadatas = kwargs.get('metadatas', [])
            ids = kwargs.get('ids', [])
            
            # Get or create collection
            if collection_name in self.collections:
                collection = self.collections[collection_name]
            else:
                try:
                    collection = self.client.get_collection(name=collection_name)
                    self.collections[collection_name] = collection
                except Exception:
                    # Create collection without embedding function
                    if not self.create_collection(collection_name, modality):
                        return False
                    collection = self.collections.get(collection_name)
                    if collection is None:
                        return False
                
            enhanced_metadatas = []
            for metadata in metadatas:
                enhanced_metadata = metadata.copy()
                
                if collection_name == "smartdoc_document_types" and "entities" in enhanced_metadata and isinstance(enhanced_metadata["entities"], list):
                    enhanced_metadata["entities"] = json.dumps(enhanced_metadata["entities"])
                elif collection_name not in ["smartdoc_documents", "smartdoc_classifier_images", "smartdoc_classifier_text"]:
                    enhanced_metadata.update({
                        "indexed_at": datetime.now().isoformat(),
                        "modality": "text" if modality == "multimodal" else modality
                    })
                    if modality == "multimodal":
                        enhanced_metadata.update({
                            "original_request": "multimodal",
                            "fallback_reason": "multimodal_not_supported"
                        })
                
                enhanced_metadatas.append(enhanced_metadata)
            
            if modality == "image" and images:
                # For image collections, compute embeddings manually using our embedding service
                try:
                    from api.services.embedding import embedding_service
                    
                    # Compute embeddings for each image
                    computed_embeddings = []
                    for image in images:
                        result = embedding_service.generate_image_embedding(image)
                        if result.get('success') and 'embedding' in result:
                            computed_embeddings.append(result['embedding'])
                        else:
                            print(f"Failed to compute embedding for image: {result.get('error', 'Unknown error')}")
                            return False
                    
                    # Add with manually computed embeddings
                    collection.add(
                        embeddings=computed_embeddings,
                        metadatas=enhanced_metadatas, 
                        ids=ids
                    )
                    
                except Exception as e:
                    print(f"Error computing image embeddings: {e}")
                    return False
            else:
                if modality == "multimodal":
                    print(f"⚠️  Multimodal embeddings not supported, using text-only for collection '{collection_name}'")
                
                # Ensure all texts are strings
                validated_texts = []
                for text in texts:
                    if isinstance(text, str):
                        validated_texts.append(text)
                    elif text is None:
                        validated_texts.append("")
                    else:
                        validated_texts.append(str(text))
                
                collection.add(documents=validated_texts, metadatas=enhanced_metadatas, ids=ids)
            
            return True
        except Exception as e:
            print(f"Failed to index in {collection_name}: {e}")
            return False
    
    def query(self, **kwargs) -> Dict[str, Any]:
        """Query a collection."""
        try:
            collection_name = kwargs.get('collection_name')
            modality = kwargs.get('modality', 'text')
            query_text = kwargs.get('query_text')
            query_image = kwargs.get('query_image')
            n_results = kwargs.get('n_results', 10)
            include = kwargs.get('include', ["metadatas", "distances"])
            
            # Get or create collection
            if collection_name in self.collections:
                collection = self.collections[collection_name]
            else:
                try:
                    collection = self.client.get_collection(name=collection_name)
                    self.collections[collection_name] = collection
                except Exception:
                    if not self.create_collection(collection_name, modality):
                        return {"success": False, "error": "Collection not found"}
                    collection = self.collections.get(collection_name)
                    if collection is None:
                        return {"success": False, "error": "Collection not found"}
                
            if modality == "image" and query_image is not None:
                results = collection.query(query_images=[query_image], n_results=n_results, include=include)
            else:
                results = collection.query(query_texts=[query_text], n_results=n_results, include=include)
            
            return {"success": True, "results": results, "collection": collection_name, "modality": modality}
        except Exception as e:
            return {"success": False, "error": str(e), "collection": collection_name, "modality": modality}