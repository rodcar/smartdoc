import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import serializers
from django.core.files.uploadedfile import UploadedFile
from drf_spectacular.utils import extend_schema
from .services.analysis import analysis_service


class DocumentAnalysisRequestSerializer(serializers.Serializer):
    image = serializers.ImageField(required=True)


class DocumentAnalysisResponseSerializer(serializers.Serializer):
    document_type = serializers.CharField()
    entities = serializers.DictField()


class ErrorResponseSerializer(serializers.Serializer):
    error = serializers.CharField()


class DocumentAnalysisView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    @extend_schema(
        operation_id='analyze_document',
        summary='Analyze Document',
        request=DocumentAnalysisRequestSerializer,
        responses={
            200: DocumentAnalysisResponseSerializer,
            400: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        tags=['Document Analysis']
    )
    def post(self, request):
        """
        Identify document type and extract relevant entities from an image.

        Args:
            request: HTTP request with 'image' field (JPG/JPEG format)

        Returns:
            Response: JSON response with analysis results or error message
        """
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
            supported_formats = ['.jpg', '.jpeg']
            
            if file_extension not in supported_formats:
                return Response(
                    {"error": f"Unsupported image format: {file_extension}. Supported formats: {', '.join(supported_formats)}"}, 
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
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {"error": f"Document analysis request failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )