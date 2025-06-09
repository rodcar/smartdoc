# Embedding Service

This module provides embedding functionality for multimodal document processing, supporting both text and image embeddings.

## Common Issues and Solutions

### PyTorch Meta Tensor Issue

**Problem**: You may encounter the error: 
```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
```

**Root Cause**: This happens when PyTorch uses meta tensors (placeholders without data) for memory-efficient model initialization, but the embedding libraries (SentenceTransformer/OpenCLIP) try to move models to GPU/MPS using the old `to()` method instead of `to_empty()`.

**Solutions Applied**:

1. **Environment Variables**: Set `TORCH_FORCE_DISABLE_META=true` to disable meta tensor usage
2. **CPU-First Initialization**: Initialize models on CPU first, then move to target device if needed
3. **Fallback Strategies**: Multiple fallback approaches if initial initialization fails
4. **Device Management**: Proper handling of device transitions

**Manual Fix**: If you encounter this error, you can set the environment variable before running:
```bash
export TORCH_FORCE_DISABLE_META=true
python your_script.py
```

Or in Python:
```python
import os
os.environ['TORCH_FORCE_DISABLE_META'] = 'true'
# Then import your embedding modules
```

### Performance Notes

- The fix may force CPU-only embedding generation in some cases for stability
- This is a temporary workaround until the underlying libraries properly support PyTorch's meta tensor functionality
- For production systems, consider using dedicated GPU instances with consistent PyTorch versions

## Providers

Currently supported providers:
- **ChromaDB Provider**: Uses OpenCLIP for images and SentenceTransformer for text
- More providers can be added by implementing the `EmbeddingProvider` interface

## Usage

```python
from api.services.embedding.embedding_service import EmbeddingService

service = EmbeddingService()

# Generate text embedding
text_result = service.generate_text_embedding("Your text here", "document.pdf")

# Generate image embedding  
image_result = service.generate_image_embedding("/path/to/image.jpg")

# Generate both embeddings
dual_result = service.generate_dual_embeddings(
    "/path/to/image.jpg", 
    "Extracted text", 
    classification_result
)
``` 