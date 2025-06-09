from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any
from django.conf import settings

from api.services.ocr.ocr_providers import AVAILABLE_PROVIDERS as OCR_PROVIDERS
from api.services.llm.llm_providers import AVAILABLE_PROVIDERS as LLM_PROVIDERS
from api.data.vectordb_providers import AVAILABLE_PROVIDERS as VECTORDB_PROVIDERS
from api.services.embedding.embedding_providers import AVAILABLE_PROVIDERS as EMBEDDING_PROVIDERS

class ValidationError(Exception):
    """Custom exception for configuration validation errors"""
    def __init__(self, message: str, error_type: str = "validation"):
        super().__init__(message)
        self.message = message
        self.error_type = error_type


@dataclass
class ProcessingConfig:
    """Configuration class for document processing pipeline with built-in validation"""
    folder_path: str
    ocr_provider: Optional[str] = None
    llm_provider: Optional[str] = None
    vectordb_provider: Optional[str] = None
    text_embedding_provider: Optional[str] = None
    image_embedding_provider: Optional[str] = None
    max_workers: Optional[int] = None
    output_json: bool = False

    def validate(self) -> None:
        """Validate all configuration parameters"""
        self._validate_folder_path()
        self._validate_providers()

    def _validate_folder_path(self) -> None:
        """Validate that the folder path exists and is a directory"""
        path = Path(self.folder_path)
        
        # Check if folder path exists
        if not path.exists():
            raise ValidationError(
                f"The specified folder path does not exist: '{self.folder_path}'. "
                f"Please check the path and ensure it exists on your file system.",
                "path_not_found"
            )
        
        # Check if folder path is a directory
        if not path.is_dir():
            raise ValidationError(
                f"The specified path is not a directory: '{self.folder_path}'. "
                f"Please provide a valid directory path containing the documents to process.",
                "invalid_directory"
            )

    def _validate_providers(self) -> None:
        """Validate all specified providers against available provider registries"""
        # Define which providers need validation and their corresponding provider classes
        providers_to_validate = [
            ('ocr', self.ocr_provider, OCR_PROVIDERS),
            ('llm', self.llm_provider, LLM_PROVIDERS),
            ('vectordb', self.vectordb_provider, VECTORDB_PROVIDERS),
            ('text_embedding', self.text_embedding_provider, EMBEDDING_PROVIDERS),
            ('image_embedding', self.image_embedding_provider, EMBEDDING_PROVIDERS)
        ]
        
        # Get default provider from settings or use 'default' as fallback
        default_provider = getattr(settings, 'DEFAULT_PROVIDER', 'default')
        
        # Check each provider configuration
        for provider_type, provider_name, available_providers in providers_to_validate:
            # Skip validation if no provider specified or using default
            if not provider_name or provider_name == default_provider:
                continue
                
            # Get list of available provider names by removing 'Provider' suffix
            available_names = [p.__name__.lower().replace('provider', '') for p in available_providers]
            
            # Raise error if specified provider not in available providers list
            if provider_name not in available_names:
                raise ValidationError(
                    f"The specified {provider_type} provider '{provider_name}' is not available. "
                    f"Please choose from the following supported providers: {', '.join(available_names)}. "
                    f"You can also use '{default_provider}' to use the system default provider.",
                    "invalid_provider"
                )

    def setup_environment(self) -> None:
        """Set up configuration from Django settings"""
        default_provider = getattr(settings, 'DEFAULT_PROVIDER', 'default')
        default_max_workers = getattr(settings, 'DEFAULT_MAX_WORKERS', 4)
        
        # Map instance attributes to their corresponding Django settings
        provider_settings = {
            'ocr_provider': 'OCR_PROVIDER',
            'llm_provider': 'LLM_PROVIDER',
            'vectordb_provider': 'VECTORDB_PROVIDER',
            'text_embedding_provider': 'TEXT_EMBEDDING_PROVIDER',
            'image_embedding_provider': 'IMAGE_EMBEDDING_PROVIDER'
        }
        
        # Set default providers from settings if not explicitly specified
        for attr, setting in provider_settings.items():
            if getattr(self, attr) is None:
                setattr(self, attr, getattr(settings, setting, default_provider))
        
        # Set max workers from settings if not set or invalid
        if self.max_workers is None or self.max_workers <= 0:
            self.max_workers = getattr(settings, 'MAX_CONCURRENT_TASKS', default_max_workers)

    @classmethod
    def from_options(cls, options: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration object from command line options"""
        folder_path = options.get('folder_path')
        
        if not folder_path:
            raise ValidationError(
                "No folder path provided. Please specify a path to the directory containing documents to process. "
                "Example: python manage.py process_documents /path/to/your/documents",
                "missing_folder_path"
            )
        
        return cls(
            folder_path=folder_path,
            ocr_provider=options.get('ocr'),
            llm_provider=options.get('llm'),
            vectordb_provider=options.get('vectordb'),
            text_embedding_provider=options.get('text_embedding'),
            image_embedding_provider=options.get('image_embedding'),
            max_workers=options.get('max_workers'),
            output_json=options.get('json', False)
        )