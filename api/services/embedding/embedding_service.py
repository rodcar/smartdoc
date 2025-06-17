"""
Embedding service for managing multimodal document embeddings.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

from .embedding_providers import EmbeddingProvider, AVAILABLE_PROVIDERS


class EmbeddingService:
    """
    Service that manages different embedding providers and provides high-level
    embedding functionality for multimodal documents.
    
    This service handles individual document processing. Batch processing and
    concurrency management is handled at the task layer.
    """
    
    # Per-worker singleton providers
    _text_embedding_provider: Optional[EmbeddingProvider] = None
    _image_embedding_provider: Optional[EmbeddingProvider] = None
    
    def __init__(self, auto_load_providers: bool = True):
        self.providers: List[EmbeddingProvider] = []
        self.default_provider: Optional[EmbeddingProvider] = None
        
        if auto_load_providers:
            self.load_available_providers()
    
    def load_available_providers(self):
        """Load all available and properly configured providers."""
        for provider_class in AVAILABLE_PROVIDERS:
            try:
                provider = provider_class()
                if provider.is_available():
                    self.providers.append(provider)
                    if self.default_provider is None:
                        self.default_provider = provider
                    print(f"✅ Loaded embedding provider: {provider.name}")
                else:
                    print(f"Provider {provider_class.__name__} is not available")
            except Exception as e:
                print(f"Warning: Could not load {provider_class.__name__}: {e}")
    
    @classmethod
    def init_text_embedding_provider(cls, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize per-worker singleton text embedding provider.
        
        Args:
            provider_name: Specific provider to initialize (optional)
            
        Returns:
            Dictionary containing initialization result
        """
        if cls._text_embedding_provider is not None:
            return {
                'success': True,
                'action': 'already_initialized',
                'provider_name': cls._text_embedding_provider.name
            }
        
        try:
            # Find and initialize the provider
            for provider_class in AVAILABLE_PROVIDERS:
                provider = provider_class()
                if provider.is_available() and 'text' in provider.supported_modalities:
                    if provider_name is None or provider.name == provider_name:
                        cls._text_embedding_provider = provider
                        return {
                            'success': True,
                            'action': 'initialized',
                            'provider_name': provider.name
                        }
            
            return {
                'success': False,
                'error': f'No available text embedding provider found' + (f' for {provider_name}' if provider_name else '')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to initialize text embedding provider: {str(e)}'
            }
    
    @classmethod
    def init_image_embedding_provider(cls, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Initialize per-worker singleton image embedding provider.
        
        Args:
            provider_name: Specific provider to initialize (optional)
            
        Returns:
            Dictionary containing initialization result
        """
        if cls._image_embedding_provider is not None:
            return {
                'success': True,
                'action': 'already_initialized',
                'provider_name': cls._image_embedding_provider.name
            }
        
        try:
            # Find and initialize the provider
            for provider_class in AVAILABLE_PROVIDERS:
                provider = provider_class()
                if provider.is_available() and 'image' in provider.supported_modalities:
                    if provider_name is None or provider.name == provider_name:
                        cls._image_embedding_provider = provider
                        return {
                            'success': True,
                            'action': 'initialized',
                            'provider_name': provider.name
                        }
            
            return {
                'success': False,
                'error': f'No available image embedding provider found' + (f' for {provider_name}' if provider_name else '')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to initialize image embedding provider: {str(e)}'
            }
    
    def add_provider(self, provider: EmbeddingProvider):
        """Add a provider instance."""
        if provider.is_available():
            self.providers.append(provider)
            if self.default_provider is None:
                self.default_provider = provider
        else:
            raise ValueError(f"Provider {provider.name} is not available")
    
    def set_default_provider(self, provider_name: str):
        """Set the default provider by name."""
        for provider in self.providers:
            if provider.name == provider_name:
                self.default_provider = provider
                return
        raise ValueError(f"Provider {provider_name} not found")
    
    def get_providers_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded providers."""
        return [
            {
                "name": provider.name,
                "class": provider.__class__.__name__,
                "supported_modalities": provider.supported_modalities,
                "is_default": provider == self.default_provider,
                "available": provider.is_available()
            }
            for provider in self.providers
        ]
    
    def generate_image_embedding(self, 
                                image: Union[str, np.ndarray, Image.Image],
                                provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an embedding for an image.
        
        Args:
            image: Image to embed (path, numpy array, or PIL Image)
            provider_name: Specific provider to use (optional)
            
        Returns:
            Dictionary containing embedding result and metadata
        """
        # Use app-level singleton provider first (fastest)
        if provider_name is None:
            try:
                from .startup import get_app_image_embedding_provider
                app_provider = get_app_image_embedding_provider()
                if app_provider is not None:
                    return app_provider.generate_image_embedding(image)
            except ImportError:
                pass  # startup module not available, continue with other providers
        
        # Use class-level singleton provider if available and no specific provider requested
        if provider_name is None and self._image_embedding_provider is not None:
            return self._image_embedding_provider.generate_image_embedding(image)
        
        # Fall back to _get_provider for other cases
        provider = self._get_provider(provider_name)
        return provider.generate_image_embedding(image)
    
    def generate_text_embedding(self, 
                               text: str,
                               source_path: str = "",
                               provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an embedding for text.
        
        Args:
            text: Text content to embed
            source_path: Optional source path for metadata
            provider_name: Specific provider to use (optional)
            
        Returns:
            Dictionary containing embedding result and metadata
        """
        # Use app-level singleton provider first (fastest)
        if provider_name is None:
            try:
                from .startup import get_app_text_embedding_provider
                app_provider = get_app_text_embedding_provider()
                if app_provider is not None:
                    return app_provider.generate_text_embedding(text, source_path)
            except ImportError:
                pass  # startup module not available, continue with other providers
        
        # Use class-level singleton provider if available and no specific provider requested
        if provider_name is None and self._text_embedding_provider is not None:
            return self._text_embedding_provider.generate_text_embedding(text, source_path)
        
        # Fall back to _get_provider for other cases
        provider = self._get_provider(provider_name)
        return provider.generate_text_embedding(text, source_path)
    

    

    
    def save_embeddings_to_file(self,
                               embedding_results: List[Dict[str, Any]],
                               output_dir: str,
                               filename_prefix: str = "embeddings") -> Dict[str, Any]:
        """
        Save embedding results to JSON files.
        
        Args:
            embedding_results: List of embedding results
            output_dir: Output directory for saving files
            filename_prefix: Prefix for output filenames
            
        Returns:
            Summary of save operations
        """
        if not embedding_results:
            return {'saved_files': 0, 'errors': 0, 'total_processed': 0}
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare separate files for different embedding types
        image_embeddings = []
        text_embeddings = []
        combined_embeddings = []
        
        saved_files = 0
        errors = 0
        
        try:
            # Process and organize embedding results
            for result in embedding_results:
                if result.get('success', False):
                    combined_embeddings.append(result)
                    
                    # Extract individual embeddings
                    if result.get('image_embedding') and result['image_embedding'].get('success'):
                        # Convert numpy arrays to lists for JSON serialization
                        img_result = result['image_embedding'].copy()
                        if img_result.get('embedding') is not None:
                            embedding = img_result['embedding']
                            if hasattr(embedding, 'tolist'):
                                img_result['embedding'] = embedding.tolist()
                        image_embeddings.append(img_result)
                    
                    if result.get('text_embedding') and result['text_embedding'].get('success'):
                        # Convert numpy arrays to lists for JSON serialization
                        txt_result = result['text_embedding'].copy()
                        if txt_result.get('embedding') is not None:
                            embedding = txt_result['embedding']
                            if hasattr(embedding, 'tolist'):
                                txt_result['embedding'] = embedding.tolist()
                        text_embeddings.append(txt_result)
                else:
                    errors += 1
            
            # Save image embeddings
            if image_embeddings:
                image_file = output_path / f"{filename_prefix}_image_embeddings.json"
                with open(image_file, 'w', encoding='utf-8') as f:
                    json.dump(image_embeddings, f, indent=2, default=str)
                print(f"Saved {len(image_embeddings)} image embeddings to {image_file}")
                saved_files += 1
            
            # Save text embeddings
            if text_embeddings:
                text_file = output_path / f"{filename_prefix}_text_embeddings.json"
                with open(text_file, 'w', encoding='utf-8') as f:
                    json.dump(text_embeddings, f, indent=2, default=str)
                print(f"Saved {len(text_embeddings)} text embeddings to {text_file}")
                saved_files += 1
            
            # Save combined metadata (without actual embedding vectors to save space)
            combined_metadata = []
            for result in combined_embeddings:
                metadata = {
                    'document_id': result.get('document_id'),
                    'image_path': result.get('image_path'),
                    'has_image_embedding': result.get('has_image_embedding', False),
                    'has_text_embedding': result.get('has_text_embedding', False),
                    'classification': result.get('classification'),
                    'processing_time': result.get('total_processing_time_seconds'),
                    'created_at': result.get('created_at')
                }
                if result.get('image_embedding'):
                    metadata['image_embedding_dim'] = result['image_embedding'].get('embedding_dimension')
                if result.get('text_embedding'):
                    metadata['text_embedding_dim'] = result['text_embedding'].get('embedding_dimension')
                    metadata['text_length'] = result['text_embedding'].get('text_length')
                combined_metadata.append(metadata)
            
            if combined_metadata:
                metadata_file = output_path / f"{filename_prefix}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_metadata, f, indent=2, default=str)
                print(f"Saved metadata for {len(combined_metadata)} documents to {metadata_file}")
                saved_files += 1
            
            # Save processing summary
            summary = {
                'total_processed': len(embedding_results),
                'successful_embeddings': len(combined_embeddings),
                'image_embeddings_count': len(image_embeddings),
                'text_embeddings_count': len(text_embeddings),
                'errors': errors,
                'saved_files': saved_files,
                'output_directory': str(output_path)
            }
            
            summary_file = output_path / f"{filename_prefix}_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Saved processing summary to {summary_file}")
            saved_files += 1
            
            return summary
            
        except Exception as e:
            print(f"Error saving embedding results: {str(e)}")
            return {
                'saved_files': saved_files,
                'errors': errors + 1,
                'total_processed': len(embedding_results),
                'error': str(e)
            }
    
    def _get_provider(self, provider_name: Optional[str] = None) -> EmbeddingProvider:
        """Get provider by name or return default."""
        # First check for class-level singleton providers
        if provider_name is None:
            # Return singleton image provider if available, otherwise text provider
            if self._image_embedding_provider is not None:
                return self._image_embedding_provider
            elif self._text_embedding_provider is not None:
                return self._text_embedding_provider
            elif self.default_provider is not None:
                return self.default_provider
            else:
                raise ValueError("No providers available")
        
        # Check if requested provider is one of the singleton providers
        if (self._image_embedding_provider is not None and 
            self._image_embedding_provider.name == provider_name):
            return self._image_embedding_provider
        
        if (self._text_embedding_provider is not None and 
            self._text_embedding_provider.name == provider_name):
            return self._text_embedding_provider
        
        # Fall back to instance providers
        for provider in self.providers:
            if provider.name == provider_name:
                return provider
        
        raise ValueError(f"Provider {provider_name} not found")
    



    def _initialize_functions(self):
        """Initialize embedding functions."""
        if self._text_embedding_provider is not None:
            self._text_embedding_provider._initialize_functions()
        if self._image_embedding_provider is not None:
            self._image_embedding_provider._initialize_functions()

# Module-level instance creation moved to __init__.py for lazy loading 