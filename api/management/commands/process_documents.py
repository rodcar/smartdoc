import json
from typing import Dict, Any

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from api.management.commands.process_documents_config import ProcessingConfig, ValidationError

from api.pipelines.flows.document_processing_flow import document_processing_flow
from api.services.ocr.ocr_providers import AVAILABLE_PROVIDERS as OCR_PROVIDERS
from api.services.llm.llm_providers import AVAILABLE_PROVIDERS as LLM_PROVIDERS
from api.data.vectordb_providers import AVAILABLE_PROVIDERS as VECTORDB_PROVIDERS
from api.services.embedding.embedding_providers import AVAILABLE_PROVIDERS as EMBEDDING_PROVIDERS


class Command(BaseCommand):
    """Pipeline for OCR, document classification, and entity extraction"""
    help = 'Pipeline for OCR, document classification, and entity extraction'

    def add_arguments(self, parser) -> None:
        """Add command line arguments for document processing pipeline"""
        parser.add_argument(
            'folder_path',
            type=str,
            nargs='?',  # Make folder_path optional when using --list-providers
            help='Path to the folder containing images to process'
        )
        
        parser.add_argument(
            '--list-providers',
            action='store_true',
            help='List all available services providers'
        )

        parser.add_argument(
            '--ocr',
            type=str,
            metavar='PROVIDER',
            help='OCR provider to use for text extraction',
            default=None
        )

        parser.add_argument(
            '--llm',
            type=str,
            metavar='PROVIDER',
            help='LLM provider to use for document classification and entity extraction',
            default=None
        )

        parser.add_argument(
            '--vectordb',
            type=str,
            metavar='PROVIDER',
            help='Vector database provider to use for document storage and retrieval',
            default=None
        )

        parser.add_argument(
            '--text-embedding',
            type=str,
            metavar='PROVIDER',
            help='Text embedding provider to use for text vectorization',
            default=None
        )

        parser.add_argument(
            '--image-embedding',
            type=str,
            metavar='PROVIDER',
            help='Image embedding provider to use for image vectorization',
            default=None
        )

        parser.add_argument(
            '--max-workers',
            type=int,
            help=f'Maximum number of concurrent processing tasks',
            default=None
        )
        
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output results in JSON format'
        )

    def handle(self, *args, **options) -> None:
        """Main command handler for document processing pipeline"""
        as_json = options.get('json', False)
        # Handle list providers request
        if options.get('list_providers'):
            self._list_providers(as_json)
            return

        # Create and validate configuration
        try:
            config = ProcessingConfig.from_options(options)
            config.validate()
        except ValidationError as e:
            self._handle_error(str(e), as_json)
            return

        # Execute document processing pipeline
        self._run_pipeline(config)

    def _run_pipeline(self, config: ProcessingConfig) -> None:
        """Execute the document processing pipeline"""
        config.setup_environment()
        
        if not config.output_json:
            self._display_pipeline_start(config)
        
        try:
            result = document_processing_flow(
                folder_path=config.folder_path,
                ocr_provider=config.ocr_provider,
                llm_provider=config.llm_provider,
                vectordb_provider=config.vectordb_provider,
                text_embedding_provider=config.text_embedding_provider,
                image_embedding_provider=config.image_embedding_provider,
                max_workers=config.max_workers
            )
            self._display_pipeline_results(result, config.output_json)
            
        except Exception as e:
            self._handle_error(f"Document processing pipeline failed: {str(e)}", config.output_json)

    def _display_pipeline_start(self, config: ProcessingConfig) -> None:
        """Display pipeline start information"""
        separator_line = getattr(settings, 'SEPARATOR_LINE', "-" * 50)
        
        self.stdout.write(self.style.SUCCESS("🚀 Starting document processing pipeline..."))
        self.stdout.write(
            f"""Input path: {config.folder_path}
OCR: {config.ocr_provider or 'default'}, LLM: {config.llm_provider or 'default'}, VectorDB: {config.vectordb_provider or 'default'}, Text Embedding: {config.text_embedding_provider or 'default'}, Image Embedding: {config.image_embedding_provider or 'default'}
Max concurrent workers: {config.max_workers}""")
        self.stdout.write(separator_line)

    def _display_pipeline_results(self, result: Dict[str, Any], as_json: bool = False) -> None:
        """Display processing results with detailed statistics and performance metrics"""
        if as_json:
            self.stdout.write(json.dumps(result, indent=2))
            return

        if not result.get('success', False):
            self.stdout.write(self.style.ERROR(f"Pipeline failed: {result.get('error', 'Unknown error')}"))
            return

        separator_line = getattr(settings, 'SEPARATOR_LINE', "-" * 50)
        
        self.stdout.write(self.style.SUCCESS("Document Processing Pipeline completed successfully!"))
        self.stdout.write(separator_line)
        
        processing_summary = result.get('processing_summary', {})
        
        # Show warning for failed extractions
        failed_count = processing_summary.get('failed_extractions', 0)
        if failed_count > 0:
            self.stdout.write(f"⚠️ Warning: {failed_count} document(s) failed to process completely")
        
        self.stdout.write(separator_line)
        self.stdout.write(self.style.SUCCESS("✅ Document Processing Pipeline completed!"))

    def _list_providers(self, as_json: bool = False) -> None:
        """List all available services providers"""
        try:
            # Map service types to their provider class names
            providers_list = {
                'ocr': [p.__name__ for p in OCR_PROVIDERS],
                'llm': [p.__name__ for p in LLM_PROVIDERS],
                'vectordb': [p.__name__ for p in VECTORDB_PROVIDERS],
                'embedding': [p.__name__ for p in EMBEDDING_PROVIDERS]
            }
            
            if as_json:
                self.stdout.write(json.dumps(providers_list, indent=2))
            else:
                separator_line = getattr(settings, 'SEPARATOR_LINE', "-" * 50)
                
                self.stdout.write("Available Providers:")
                self.stdout.write(separator_line)
                for service_type, providers in providers_list.items():
                    self.stdout.write(f"{service_type}: {', '.join(providers) or 'None'}")
                
        except Exception as e:
            self._handle_error(f"Failed to list providers: {str(e)}", as_json)

    def _handle_error(self, error_msg: str, as_json: bool = False) -> None:
        """Handle errors with consistent formatting"""
        if as_json:
            error_result = {
                'success': False,
                'error': error_msg
            }
            self.stdout.write(json.dumps(error_result, indent=2))
        else:
            raise CommandError(error_msg)