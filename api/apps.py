from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        """Initialize app-level services when Django starts."""
        import os
        if os.environ.get('RUN_MAIN') or os.environ.get('DJANGO_SETTINGS_MODULE'):
            try:
                from .services.embedding.startup import initialize_app_embeddings
                initialize_app_embeddings()
                print("📱 SmartDoc app ready")
            except Exception as e:
                print(f"❌ SmartDoc app failed: {e}")
