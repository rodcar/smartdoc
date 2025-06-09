import os
from django.apps import AppConfig
from .services.embedding.embedding_providers import AVAILABLE_PROVIDERS

# App-level singleton providers
_text_provider = None
_image_provider = None
_initialized = False


def initialize_app_embeddings():
    """Initialize app-level embedding providers at Django startup."""
    global _text_provider, _image_provider, _initialized
    
    if _initialized:
        return
    
    _initialized = True
    
    if os.environ.get('SMARTDOC_PRELOAD_EMBEDDINGS', 'true').lower() != 'true':
        print("🔄 Embedding preloading disabled")
        return
    
    print("🚀 Initializing app-level embeddings...")
    
    try:
        # Initialize providers
        for provider_class in AVAILABLE_PROVIDERS:
            provider = provider_class()
            if not provider.is_available():
                continue
                
            provider._initialize_functions()
            
            if not _text_provider and 'text' in provider.supported_modalities:
                _text_provider = provider
                print(f"✅ Text: {provider.name}")
            
            if not _image_provider and 'image' in provider.supported_modalities:
                _image_provider = provider
                print(f"✅ Image: {provider.name}")
        
        print("🎯 App embeddings ready")
        
    except Exception as e:
        print(f"⚠️ Embedding init failed: {e}")


def get_app_text_embedding_provider():
    """Get app-level text provider."""
    return _text_provider


def get_app_image_embedding_provider():
    """Get app-level image provider."""
    return _image_provider


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        """Initialize app-level services when Django starts."""
        if os.environ.get('RUN_MAIN') or os.environ.get('DJANGO_SETTINGS_MODULE'):
            try:
                initialize_app_embeddings()
                print("📱 SmartDoc app ready")
            except Exception as e:
                print(f"❌ SmartDoc app failed: {e}")
