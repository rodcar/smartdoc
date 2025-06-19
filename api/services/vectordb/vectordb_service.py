"""
VectorDB service for document indexing and storage.
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from datetime import datetime
import json

from api.data.vectordb_providers import AVAILABLE_PROVIDERS


class VectorDBService:
    """Simple service for vector database operations."""
    
    def __init__(self, auto_load_providers: bool = False):
        self.providers = {}
        self.default_provider = None
        self.db_path: Optional[str] = None
        self._providers_loaded = False
        
        if auto_load_providers:
            self.load_available_providers()

    def load_available_providers(self):
        """Load all available providers."""
        if self._providers_loaded:
            return
            
        for provider_class in AVAILABLE_PROVIDERS:
            try:
                provider = provider_class()
                self.providers[provider.name] = provider
                if self.default_provider is None:
                    self.default_provider = provider
                print(f"✅ Loaded vectordb provider: {provider.name}")
            except Exception as e:
                print(f"Warning: Could not load {provider_class.__name__}: {e}")
        
        self._providers_loaded = True
    
    def _ensure_providers_loaded(self):
        """Ensure providers are loaded."""
        if not self._providers_loaded:
            self.load_available_providers()
    
    def set_provider(self, provider_name: str) -> bool:
        """Set the active provider by name."""
        self._ensure_providers_loaded()
        
        if provider_name in self.providers:
            self.default_provider = self.providers[provider_name]
            print(f"✅ Set active provider to: {provider_name}")
            return True
        else:
            available_providers = list(self.providers.keys())
            print(f"❌ Provider '{provider_name}' not available. Available providers: {available_providers}")
            return False
      
    def get_providers_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded providers."""
        self._ensure_providers_loaded()
        return [
            {
                "name": provider.name,
                "class": provider.__class__.__name__,
                "is_default": provider == self.default_provider,
                "available": provider.is_available()
            }
            for provider in self.providers.values()
        ]

    def check_database_exists(self, **kwargs) -> bool:
        """Check if vector database exists."""
        return self.default_provider.exists(**kwargs)

    def create_and_initialize_database(self, db_path: str, collection_name: str = "smartdoc_documents") -> bool:
        """Create and initialize a new vector database."""
        try:
            success = self.default_provider.initialize(db_path=db_path)
            if not success:
                raise Exception("Failed to initialize database")
            self.db_path = db_path
            return True
        except Exception as e:
            raise Exception(f"Failed to create and initialize database: {e}")
       
    def initialize_database(self, **kwargs) -> bool:
        """Initialize the vector database."""
        self._ensure_providers_loaded()
        if not self.default_provider:
            print("No vector database provider available")
            return False
            
        success = self.default_provider.initialize(**kwargs)
        if not success:
            print(f"❌ Failed to initialize vector database")
            
        return success

    def create_document_id(self, file_path: str) -> str:
        """Create a unique document ID based on file path."""
        return hashlib.md5(file_path.encode()).hexdigest()
    
    def image_to_base64(self, image_path: str) -> Optional[str]:
        """Convert an image file to base64 string."""
        try:
            import base64
            
            # Ensure proper path handling for special characters
            resolved_path = str(Path(image_path).resolve())
            with open(resolved_path, "rb") as image_file:
                image_data = image_file.read()
                base64_string = base64.b64encode(image_data).decode('utf-8')
                
                file_extension = Path(resolved_path).suffix.lower()
                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg', 
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp'
                }
                
                mime_type = mime_types.get(file_extension, 'image/jpeg')
                return f"data:{mime_type};base64,{base64_string}"
                
        except Exception as e:
            print(f"Failed to convert image to base64 {image_path}: {e}")
            return None
    
    def load_image_as_array(self, image_path: str) -> Optional[np.ndarray]:
        """Load an image and convert it to a numpy array."""
        try:
            # Ensure proper path handling for special characters
            from pathlib import Path
            resolved_path = str(Path(image_path).resolve())
            with Image.open(resolved_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image_array = np.array(img)
                return image_array
        except Exception as e:
            print(f"⚠️  Error loading image {os.path.basename(image_path)}: {str(e)}")
            return None
    
    def index_documents(self,
                            documents: List[Dict[str, Any]],
                            collection_name: str = "smartdoc_documents") -> Dict[str, Any]:
        """Index a batch of documents with text embeddings and save document types."""
        if not self.default_provider:
            return {
                'success': False,
                'error': 'No vector database provider available',
                'indexed_documents': 0,
                'failed_count': len(documents)
            }

        try:
            # Prepare batch data
            valid_documents = []
            document_types_entities = {}
            
            for doc in documents:
                image_path = doc.get('image_path')
                extracted_text = doc.get('extracted_text', '')
                document_type = doc.get('document_type')
                
                if not image_path or not document_type:
                    continue
                    
                # Load image for base64 encoding
                image_base64 = self.image_to_base64(image_path)
                if image_base64:
                    valid_documents.append({
                        'image_path': image_path,
                        'extracted_text': extracted_text,  # Can be empty string
                        'document_type': document_type,
                        'image_base64': image_base64,
                        'extracted_entities': doc.get('extracted_entities')
                    })
                    # Collect entities for document type (avoid duplicates by entity name)
                    if document_type not in document_types_entities:
                        document_types_entities[document_type] = {}
                    entities = doc.get('extracted_entities', [])
                    if isinstance(entities, list):
                        for entity in entities:
                            if isinstance(entity, dict) and 'name' in entity and 'value' in entity and 'description' in entity:
                                entity_name = entity['name'].strip().lower()
                                if entity_name not in document_types_entities[document_type]:
                                    document_types_entities[document_type][entity_name] = entity['description'].strip()
            
            if not valid_documents:
                return {
                    'success': False,
                    'error': 'No valid documents to index',
                    'indexed_documents': 0,
                    'failed_count': len(documents)
                }
            
            # Create collections if they don't exist
            if collection_name not in self.default_provider.collections:
                if not self.default_provider.create_collection(collection_name, "text"):
                    return {
                        'success': False,
                        'error': 'Failed to create document collection',
                        'indexed_documents': 0,
                        'failed_count': len(documents)
                    }
            
            # Create document types collection
            doc_types_collection = "smartdoc_document_types"
            if doc_types_collection not in self.default_provider.collections:
                self.default_provider.create_collection(doc_types_collection, "text")
            
            # Prepare documents for indexing
            texts_to_embed = []
            metadatas = []
            document_ids = []
            
            for doc in valid_documents:
                doc_id = self.create_document_id(doc['image_path'])
                
                metadata = {
                    "type": doc['document_type'],
                    "indexed_at": datetime.now().isoformat(),
                    "ocr_text": doc['extracted_text'],  # Can be empty string
                    "base64": doc['image_base64']
                }
                
                # Add entities if available
                if doc.get('extracted_entities'):
                    metadata["entities"] = json.dumps(doc['extracted_entities'])
                
                texts_to_embed.append(doc['extracted_text'] or " ")
                metadatas.append(metadata)
                document_ids.append(doc_id)
            
            # Index documents to main collection
            success = self.default_provider.index(
                collection_name=collection_name,
                modality="text",
                texts=texts_to_embed,
                metadatas=metadatas,
                ids=document_ids
            )
            
            # Index document types to document types collection
            document_types_saved = 0
            for doc_type, entity_dict in document_types_entities.items():
                try:
                    doc_type_id = f"doctype_{doc_type}"
                    entities_list = [{"name": name, "description": desc} for name, desc in entity_dict.items()]
                    doc_type_metadata = {
                        "type": doc_type,
                        "entities": entities_list,
                        "saved_at": datetime.now().isoformat()
                    }
                    
                    type_success = self.default_provider.index(
                        collection_name=doc_types_collection,
                        modality="text",
                        texts=[doc_type],
                        metadatas=[doc_type_metadata],
                        ids=[doc_type_id]
                    )
                    
                    if type_success:
                        document_types_saved += 1
                        
                except Exception as e:
                    print(f"Warning: Failed to save document type {doc_type}: {e}")
            
            if success:
                return {
                    'success': True,
                    'indexed_documents': len(texts_to_embed),
                    'indexed_images': 0,
                    'indexed_texts': len(texts_to_embed),
                    'failed_count': len(documents) - len(valid_documents),
                    'total_documents': len(documents),
                    'collection_name': collection_name,
                    'document_types_saved': document_types_saved,
                    'document_types_failed': len(document_types_entities) - document_types_saved
                }
            else:
                return {
                    'success': False,
                    'error': 'Vector database indexing failed',
                    'indexed_documents': 0,
                    'failed_count': len(documents)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'indexed_documents': 0,
                'failed_count': len(documents)
            }

    def index_document_to_classifier_collections(self,
                                                image_path: str,
                                                extracted_text: str,
                                                document_type: str,
                                                image_collection_name: str = "smartdoc_classifier_images",
                                                text_collection_name: str = "smartdoc_classifier_text",
                                                extracted_entities: Optional[Dict[str, Any]] = None,
                                                shared_uuid: Optional[str] = None) -> Dict[str, Any]:
        """Index a document to both classifier collections."""
        import uuid as uuid_module
        
        if not self.default_provider:
            return {
                'success': False,
                'error': 'No vector database provider available',
                'file_path': image_path,
                'image_indexed': False,
                'text_indexed': False
            }

        try:
            # Load image
            image_array = self.load_image_as_array(image_path)
            if image_array is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'file_path': image_path,
                    'image_indexed': False,
                    'text_indexed': False
                }

            # Generate shared UUID
            if shared_uuid is None:
                shared_uuid = str(uuid_module.uuid4())

            # Create metadata
            metadata = {
                "uuid": shared_uuid,
                "type": document_type
            }

            results = {
                'success': False,
                'file_path': image_path,
                'shared_uuid': shared_uuid,
                'document_type': document_type,
                'image_indexed': False,
                'text_indexed': False,
                'image_error': None,
                'text_error': None
            }

            # Index to image collection
            try:
                image_success = self.default_provider.index(
                    collection_name=image_collection_name,
                    modality="image",
                    images=[image_array],
                    metadatas=[metadata.copy()],
                    ids=[f"{shared_uuid}_img"]
                )
                results['image_indexed'] = image_success
                if not image_success:
                    results['image_error'] = 'Failed to add image embedding'
            except Exception as e:
                results['image_error'] = str(e)

            # Index to text collection
            if extracted_text and extracted_text.strip():
                try:
                    text_success = self.default_provider.index(
                        collection_name=text_collection_name,
                        modality="text",
                        texts=[extracted_text],
                        metadatas=[metadata.copy()],
                        ids=[f"{shared_uuid}_txt"]
                    )
                    results['text_indexed'] = text_success
                    if not text_success:
                        results['text_error'] = 'Failed to add text embedding'
                except Exception as e:
                    results['text_error'] = str(e)
            else:
                results['text_indexed'] = False
                results['text_error'] = 'No text provided for indexing'

            # Overall success if at least one embedding was created
            results['success'] = results['image_indexed'] or results['text_indexed']
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': image_path,
                'image_indexed': False,
                'text_indexed': False
            }