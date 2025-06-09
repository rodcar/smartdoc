"""
ChromaDB embedding provider using OpenCLIP for images and SentenceTransformer for text.
"""
import os
import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from . import EmbeddingProvider

# Try to import embedding dependencies
try:
    from chromadb.utils.embedding_functions.open_clip_embedding_function import OpenCLIPEmbeddingFunction
    from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
    HAS_EMBEDDING_FUNCTIONS = True
except ImportError:
    HAS_EMBEDDING_FUNCTIONS = False


class ChromaDBProvider(EmbeddingProvider):
    """ChromaDB embedding provider using OpenCLIP + SentenceTransformer with MPS support."""
    
    def __init__(self, 
                 text_model: str = "all-MiniLM-L6-v2",
                 image_model: str = "openclip"):
        super().__init__("chromadb")
        self.text_model = text_model
        self.image_model = image_model
        self._image_embedding_function = None
        self._text_embedding_function = None
        self._initialized = False
        self._device = None
    
    def is_available(self) -> bool:
        """Check if embedding functions are available."""
        return HAS_EMBEDDING_FUNCTIONS
    
    def _get_optimal_device(self):
        """Determine the optimal device for embedding computation."""
        import torch
        
        # Check for MPS (Apple Silicon) first - best performance on macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS device works
                test_tensor = torch.tensor([1.0], device='mps')
                print("✅ Using MPS (Apple Silicon GPU) for optimal performance")
                return 'mps'
            except Exception as e:
                print(f"⚠️ MPS available but failed test: {e}, falling back to CPU")
                return 'cpu'
        
        # Check for CUDA (NVIDIA GPU) second
        elif torch.cuda.is_available():
            try:
                # Test CUDA device works
                test_tensor = torch.tensor([1.0], device='cuda')
                print("✅ Using CUDA (NVIDIA GPU) for optimal performance")
                return 'cuda'
            except Exception as e:
                print(f"⚠️ CUDA available but failed test: {e}, falling back to CPU")
                return 'cpu'
        
        # Fall back to CPU
        else:
            print("✅ Using CPU (no GPU acceleration available)")
            return 'cpu'
    
    def _initialize_functions(self):
        """Initialize embedding functions if not already initialized."""
        if self._initialized or not HAS_EMBEDDING_FUNCTIONS:
            return
        
        try:
            # Initialize with proper device and tensor handling
            self._initialize_with_device_management()
            self._initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding functions: {str(e)}")
    
    def _initialize_with_device_management(self):
        """Initialize embedding functions with proper PyTorch device management."""
        import torch
        import os
        
        # Set environment variables to handle meta tensors properly
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        # Disable meta tensor usage to avoid the device movement issue
        os.environ.setdefault('TORCH_FORCE_DISABLE_META', 'true')
        
        # Determine the best available device (MPS > CUDA > CPU)
        self._device = self._get_optimal_device()
        print(f"🎯 Selected device: {self._device}")
        
        try:
            # Initialize with proper meta tensor handling
            self._initialize_with_meta_tensor_fix()
            
        except Exception as e:
            # If all fails, try one more time with torch settings
            print(f"⚠️ Initialization failed, trying with modified torch settings...")
            self._initialize_with_torch_settings()
    
    def _initialize_with_meta_tensor_fix(self):
        """Initialize embedding functions with proper meta tensor handling."""
        import torch
        
        # Disable meta tensor functionality temporarily
        original_meta_enabled = getattr(torch, '_C', {}).get('_meta', None)
        
        try:
            # Force disable meta tensor mode during initialization
            with torch.device('cpu'):  # Initialize on CPU first
                # Initialize OpenCLIP for image embeddings
                try:
                    self._image_embedding_function = OpenCLIPEmbeddingFunction()
                    print("✅ OpenCLIP embedding function initialized successfully")
                except Exception as e:
                    if "meta tensor" in str(e).lower():
                        print(f"⚠️ OpenCLIP meta tensor issue detected, trying CPU-only initialization...")
                        # Force CPU-only initialization
                        with torch.no_grad():
                            self._image_embedding_function = OpenCLIPEmbeddingFunction()
                    else:
                        raise e
                
                # Initialize SentenceTransformer for text embeddings
                try:
                    self._text_embedding_function = SentenceTransformerEmbeddingFunction(
                        model_name=self.text_model
                    )
                    print("✅ SentenceTransformer embedding function initialized successfully")
                except Exception as e:
                    if "meta tensor" in str(e).lower():
                        print(f"⚠️ SentenceTransformer meta tensor issue detected, trying CPU-only initialization...")
                        # Force CPU-only initialization
                        with torch.no_grad():
                            self._text_embedding_function = SentenceTransformerEmbeddingFunction(
                                model_name=self.text_model
                            )
                    else:
                        raise e
                        
        except Exception as e:
            print(f"⚠️ Meta tensor fix failed: {e}, trying fallback methods...")
            self._initialize_openclip_fallback()
            self._initialize_sentence_transformer_fallback()
    
    def _initialize_openclip_fallback(self):
        """Fallback initialization for OpenCLIP with meta tensor handling."""
        import torch
        import os
        
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Try with explicit CPU initialization and no meta tensors
            old_env = os.environ.get('TORCH_FORCE_DISABLE_META', None)
            os.environ['TORCH_FORCE_DISABLE_META'] = 'true'
            
            with torch.no_grad():
                with torch.device('cpu'):
                    self._image_embedding_function = OpenCLIPEmbeddingFunction()
                    
            # Restore environment
            if old_env is None:
                os.environ.pop('TORCH_FORCE_DISABLE_META', None)
            else:
                os.environ['TORCH_FORCE_DISABLE_META'] = old_env
                
        except Exception as e:
            # If still failing, force CPU device for stability
            self._device = 'cpu'
            with torch.no_grad():
                self._image_embedding_function = OpenCLIPEmbeddingFunction()
    
    def _initialize_sentence_transformer_fallback(self):
        """Fallback initialization for SentenceTransformer with meta tensor handling."""
        import torch
        import os
        
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Try with explicit CPU initialization and no meta tensors
            old_env = os.environ.get('TORCH_FORCE_DISABLE_META', None)
            os.environ['TORCH_FORCE_DISABLE_META'] = 'true'
            
            with torch.no_grad():
                with torch.device('cpu'):
                    self._text_embedding_function = SentenceTransformerEmbeddingFunction(
                        model_name=self.text_model
                    )
                    
            # Restore environment
            if old_env is None:
                os.environ.pop('TORCH_FORCE_DISABLE_META', None)
            else:
                os.environ['TORCH_FORCE_DISABLE_META'] = old_env
                
        except Exception as e:
            # If still failing, force CPU device for stability
            self._device = 'cpu'
            with torch.no_grad():
                self._text_embedding_function = SentenceTransformerEmbeddingFunction(
                    model_name=self.text_model
                )
    
    def _initialize_with_torch_settings(self):
        """Final fallback initialization with modified torch settings."""
        import torch
        import os
        
        # Save current settings
        old_default_dtype = torch.get_default_dtype()
        old_meta_env = os.environ.get('TORCH_FORCE_DISABLE_META', None)
        
        try:
            # Set to float32 to avoid potential precision issues
            torch.set_default_dtype(torch.float32)
            
            # Force disable meta tensors completely
            os.environ['TORCH_FORCE_DISABLE_META'] = 'true'
            
            # Disable gradient computation during initialization
            with torch.no_grad():
                # Force CPU usage for meta tensor issues
                with torch.device('cpu'):
                    # Initialize functions
                    self._image_embedding_function = OpenCLIPEmbeddingFunction()
                    self._text_embedding_function = SentenceTransformerEmbeddingFunction(
                        model_name=self.text_model
                    )
                
                # Override device selection for stability
                self._device = 'cpu'
                print("✅ Embedding functions initialized with torch settings workaround (CPU-only)")
                
        finally:
            # Restore original settings
            torch.set_default_dtype(old_default_dtype)
            if old_meta_env is None:
                os.environ.pop('TORCH_FORCE_DISABLE_META', None)
            else:
                os.environ['TORCH_FORCE_DISABLE_META'] = old_meta_env
    
    def _create_document_id(self, source: str) -> str:
        """Create a unique document ID based on source."""
        return hashlib.md5(source.encode()).hexdigest()
    
    def _load_image_as_array(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Load an image and convert it to a numpy array format expected by embedding functions.
        
        Args:
            image: Image path, numpy array, or PIL Image
            
        Returns:
            Image as numpy array
        """
        if isinstance(image, str):
            # Load from file path
            with Image.open(image) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return np.array(img)
        
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        
        elif isinstance(image, np.ndarray):
            # Already a numpy array
            return image
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def generate_image_embedding(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Generate an embedding for an image using OpenCLIP.
        
        Args:
            image: Image path, numpy array, or PIL Image
            
        Returns:
            Dictionary containing embedding result and metadata
        """
        self._initialize_functions()
        start_time = time.time()
        
        try:
            # Load image as array
            image_array = self._load_image_as_array(image)
            
            # Generate embedding using OpenCLIP
            embedding = self._image_embedding_function([image_array])
            
            # Extract the actual embedding vector (first item in the list)
            embedding_vector = embedding[0] if isinstance(embedding, list) else embedding
            
            processing_time = time.time() - start_time
            
            # Create document ID based on image source
            if isinstance(image, str):
                doc_id = self._create_document_id(image)
                source_path = image
            else:
                # For non-file inputs, create ID from array hash
                source_path = "in_memory_image"
                doc_id = self._create_document_id(str(hash(image_array.tobytes())))
            
            result = {
                'image_path': source_path,
                'document_id': doc_id,
                'embedding_id': f"img_{doc_id}",
                'embedding': embedding_vector,
                'embedding_type': 'image',
                'model': 'OpenCLIP',
                'device': self._device,
                'embedding_dimension': len(embedding_vector) if hasattr(embedding_vector, '__len__') else None,
                'processing_time_seconds': round(processing_time, 3),
                'success': True,
                'error': None,
                'created_at': time.time()
            }
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            source_path = image if isinstance(image, str) else "in_memory_image"
            doc_id = self._create_document_id(source_path)
            error_msg = f"Image embedding failed for {source_path}: {str(e)}"
            
            return {
                'image_path': source_path,
                'document_id': doc_id,
                'embedding_id': f"img_{doc_id}",
                'embedding': None,
                'embedding_type': 'image',
                'model': 'OpenCLIP',
                'device': self._device,
                'embedding_dimension': None,
                'processing_time_seconds': round(processing_time, 3),
                'success': False,
                'error': error_msg,
                'created_at': time.time()
            }
    
    def generate_text_embedding(self, text: str, source_path: str = "") -> Dict[str, Any]:
        """
        Generate an embedding for text using SentenceTransformer.
        
        Args:
            text: Text content to embed
            source_path: Optional source path for metadata
            
        Returns:
            Dictionary containing embedding result and metadata
        """
        self._initialize_functions()
        start_time = time.time()
        
        try:
            # Skip if no text
            if not text or not text.strip():
                raise ValueError("No text available for embedding")
            
            # Generate embedding using SentenceTransformer
            embedding = self._text_embedding_function([text])
            
            # Extract the actual embedding vector (first item in the list)
            embedding_vector = embedding[0] if isinstance(embedding, list) else embedding
            
            processing_time = time.time() - start_time
            doc_id = self._create_document_id(source_path or text)
            
            result = {
                'image_path': source_path,
                'document_id': doc_id,
                'embedding_id': f"txt_{doc_id}",
                'embedding': embedding_vector,
                'embedding_type': 'text',
                'model': f'SentenceTransformer-{self.text_model}',
                'device': self._device,
                'embedding_dimension': len(embedding_vector) if hasattr(embedding_vector, '__len__') else None,
                'text_content': text,
                'text_length': len(text),
                'processing_time_seconds': round(processing_time, 3),
                'success': True,
                'error': None,
                'created_at': time.time()
            }
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            doc_id = self._create_document_id(source_path or text)
            error_msg = f"Text embedding failed for {source_path}: {str(e)}"
            
            return {
                'image_path': source_path,
                'document_id': doc_id,
                'embedding_id': f"txt_{doc_id}",
                'embedding': None,
                'embedding_type': 'text',
                'model': f'SentenceTransformer-{self.text_model}',
                'device': self._device,
                'embedding_dimension': None,
                'text_content': text,
                'text_length': len(text) if text else 0,
                'processing_time_seconds': round(processing_time, 3),
                'success': False,
                'error': error_msg,
                'created_at': time.time()
            }
    
    def generate_dual_embeddings(self,
                                image: Union[str, np.ndarray, Image.Image],
                                text: str,
                                classification_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate both image and text embeddings for a single document.
        
        Args:
            image: Image to embed
            text: Text to embed
            classification_result: Optional classification result for metadata
            
        Returns:
            Dictionary containing both embedding results
        """
        start_time = time.time()
        
        try:
            # Get source path for metadata
            source_path = image if isinstance(image, str) else "in_memory_image"
            
            # Generate embeddings
            image_result = self.generate_image_embedding(image)
            text_result = self.generate_text_embedding(text, source_path) if text and text.strip() else None
            
            total_time = time.time() - start_time
            doc_id = self._create_document_id(source_path)
            
            # Combine results
            combined_result = {
                'document_id': doc_id,
                'image_path': source_path,
                'image_embedding': image_result,
                'text_embedding': text_result,
                'classification': classification_result,
                'device': self._device,
                'has_image_embedding': image_result['success'] if image_result else False,
                'has_text_embedding': text_result['success'] if text_result else False,
                'total_processing_time_seconds': round(total_time, 3),
                'success': True,
                'created_at': time.time()
            }
            
            return combined_result
            
        except Exception as e:
            total_time = time.time() - start_time
            source_path = image if isinstance(image, str) else "in_memory_image"
            error_msg = f"Dual embedding generation failed for {source_path}: {str(e)}"
            
            return {
                'document_id': self._create_document_id(source_path),
                'image_path': source_path,
                'image_embedding': None,
                'text_embedding': None,
                'classification': classification_result,
                'device': self._device,
                'has_image_embedding': False,
                'has_text_embedding': False,
                'total_processing_time_seconds': round(total_time, 3),
                'success': False,
                'error': error_msg,
                'created_at': time.time()
            }
    
    @property
    def supported_modalities(self) -> List[str]:
        """Return list of supported modalities."""
        return ['image', 'text'] 