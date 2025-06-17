"""
ChromaDB embedding provider using OpenCLIP for images and SentenceTransformer for text.
"""
import os
import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union
from PIL import Image

from .base import EmbeddingProvider

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
        if (hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available() and 
            torch.backends.mps.is_built()):
            try:
                # Test MPS device works with proper initialization
                torch.device('mps')
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
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TORCH_FORCE_DISABLE_META'] = 'true'
        
        # Determine the best available device (MPS > CUDA > CPU)
        self._device = self._get_optimal_device()
        print(f"🎯 Selected device: {self._device}")
        
        try:
            # Initialize with proper imports first, then let models use MPS
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            self._image_embedding_function = OpenCLIPEmbeddingFunction()
            self._text_embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.text_model)
            print("✅ OpenCLIP embedding function initialized successfully")
            print("✅ SentenceTransformer embedding function initialized successfully")
            self._initialized = True
            
        except Exception as e:
            print(f"⚠️ Initialization failed: {e}")
            # If initialization fails, force CPU and retry
            print("⚠️ Falling back to CPU initialization...")
            self._device = 'cpu'
            try:
                with torch.device('cpu'):
                    self._image_embedding_function = OpenCLIPEmbeddingFunction()
                    self._text_embedding_function = SentenceTransformerEmbeddingFunction(
                        model_name=self.text_model
                    )
                print("✅ Embedding functions initialized with CPU fallback")
                self._initialized = True
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize embedding functions even on CPU: {str(e2)}")
    
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
            # Load from file path - ensure proper path handling for special characters
            from pathlib import Path
            image_path = str(Path(image).resolve())
            with Image.open(image_path) as img:
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
    

    
    @property
    def name(self) -> str:
        """Return a human-readable name for this embedding provider."""
        return "chromadb"
    
    @property
    def supported_modalities(self) -> List[str]:
        """Return list of supported modalities."""
        return ['image', 'text'] 