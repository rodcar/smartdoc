import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.uploadedfile import UploadedFile
from .services.analysis import analysis_service

class DocumentAnalysisView(APIView):
    """
    API endpoint for document analysis.
    Handles HTTP requests and delegates processing to the analysis service.
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
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            supported_formats = ['.jpg', '.jpeg', '.png']
            
            if file_extension not in supported_formats:
                return Response(
                    {"error": f"Unsupported image format: {file_extension}"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Delegate to analysis service
            result = analysis_service.analyze_document(uploaded_file)
            
            if not result.get('success'):
                return Response(
                    {"error": result.get('error', 'Analysis failed')}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Return results (excluding internal success flag)
            response_data = {k: v for k, v in result.items() if k != 'success'}
            return Response(response_data)
            
        except Exception as e:
            return Response(
                {"error": f"Document analysis request failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )