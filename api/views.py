from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.uploadedfile import UploadedFile
from celery.result import AsyncResult
import os
import time
from .services.ocr import ocr_service
from .services.search import search_service
from .services.llm import llm_service

class DocumentAnalysisView(APIView):
    """
    API endpoint for comprehensive document analysis.
    Optimized with model preloading for fast response times.
    """
    
    def post(self, request):
        try:
            # Check if image was uploaded
            if 'image' not in request.FILES:
                return Response(
                    {"error": "No image file provided"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            uploaded_file: UploadedFile = request.FILES['image']
            
            # Validate image format
            import os
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            supported_formats = set()
            for provider_info in ocr_service.get_providers_info():
                supported_formats.update(provider_info['supported_formats'])
            
            if file_extension not in supported_formats:
                return Response(
                    {"error": f"Unsupported image format: {file_extension}"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Parallel processing after OCR for optimal performance
            import concurrent.futures
            import base64
            
            try:
                # Step 1: Extract text using OCR (must complete first)
                start_time = time.time()
                ocr_result = ocr_service.extract_text_with_confidence(uploaded_file.file)
                extracted_text = ocr_result['text']
                ocr_time = time.time() - start_time
                print(f"⏱️ OCR completed in {ocr_time:.2f}s")
                
                # Step 2: Parallel processing for classification and base64 preparation
                parallel_start = time.time()
                
                def run_classification():
                    """Run document classification."""
                    return search_service.classify_document(uploaded_file, ocr_text=extracted_text)
                
                def prepare_base64():
                    """Prepare base64 image for entity extraction."""
                    uploaded_file.seek(0)  # Reset file pointer
                    return base64.b64encode(uploaded_file.read()).decode('utf-8')
                
                # Execute classification and base64 preparation in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit both tasks
                    classification_future = executor.submit(run_classification)
                    base64_future = executor.submit(prepare_base64)
                    
                    # Wait for both to complete
                    classification_result = classification_future.result()
                    image_base64 = base64_future.result()
                
                parallel_time = time.time() - parallel_start
                print(f"⏱️ Parallel classification + base64 completed in {parallel_time:.2f}s")
                
                # Validate classification result
                if not classification_result.get('success') or not classification_result.get('document_type'):
                    return Response(
                        {"error": "Document type classification failed"}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                document_type = classification_result['document_type']
                
                # Step 3: Extract entities using LLM service (now with pre-prepared base64)
                entity_start = time.time()
                entity_result = llm_service.extract_entities(
                    image_base64=image_base64,
                    extracted_text=extracted_text,
                    document_type=document_type
                )
                
                if not entity_result.get('success'):
                    return Response(
                        {"error": f"Entity extraction failed: {entity_result.get('error', 'Unknown error')}"}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                entities = entity_result.get('entities', [])
                entity_time = time.time() - entity_start
                print(f"⏱️ Entity extraction completed in {entity_time:.2f}s")
                
                total_time = ocr_time + parallel_time + entity_time
                print(f"🎯 Total processing time: {total_time:.2f}s (saved ~0.5s with parallelization)")
                
                # Return results
                return Response({
                    "document_type": document_type,
                    "entities": entities
                })
                
            except Exception as e:
                return Response(
                    {"error": f"Document analysis failed: {str(e)}"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
        except Exception as e:
            return Response(
                {"error": f"Document analysis request failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    # def get(self, request):
    #     """Get information about document analysis capabilities."""
    #     return Response({
    #         "message": "Upload an image using POST request with 'image' field for comprehensive document analysis.",
    #         "response_format": {
    #             "document_type": "string - The classified document type",
    #             "entities": "array - List of entities with name and value properties"
    #         },
    #         "services_used": [
    #             "OCR for text extraction",
    #             "Dual confidence search for document classification", 
    #             "LLM service for entity extraction"
    #         ],
    #         "supported_formats": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    #     })