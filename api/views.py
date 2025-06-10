import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import serializers
from django.core.files.uploadedfile import UploadedFile
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample
from drf_spectacular.types import OpenApiTypes
from .services.analysis import analysis_service


class DocumentAnalysisRequestSerializer(serializers.Serializer):
    """Serializer for document analysis request."""
    image = serializers.ImageField(
        help_text="Image file to analyze (supported formats: JPG, JPEG, PNG)",
        required=True
    )


class DocumentAnalysisResponseSerializer(serializers.Serializer):
    """Serializer for document analysis response."""
    # Add fields based on what your analysis service returns
    # You may need to adjust these based on your actual response structure
    text_content = serializers.CharField(
        help_text="Extracted text content from the document",
        required=False
    )
    analysis_results = serializers.DictField(
        help_text="Analysis results and extracted data",
        required=False
    )
    metadata = serializers.DictField(
        help_text="Document metadata and processing information",
        required=False
    )


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses."""
    error = serializers.CharField(help_text="Error message describing what went wrong")


class DocumentAnalysisView(APIView):
    """
    API endpoint for document analysis.
    
    This endpoint accepts an image file and performs comprehensive document analysis
    including text extraction, data parsing, and content understanding.
    """
    parser_classes = (MultiPartParser, FormParser)
    
    @extend_schema(
        operation_id='analyze_document',
        summary='Analyze Document',
        description=(
            'Upload an image file to perform document analysis. '
            'The service will extract text, analyze content, and return structured data.'
        ),
        request=DocumentAnalysisRequestSerializer,
        responses={
            200: DocumentAnalysisResponseSerializer,
            400: ErrorResponseSerializer,
            500: ErrorResponseSerializer,
        },
        examples=[
            OpenApiExample(
                'Success Response',
                summary='Successful document analysis',
                description='Example of a successful document analysis response',
                value={
                    'text_content': 'Sample extracted text from the document...',
                    'analysis_results': {
                        'document_type': 'invoice',
                        'entities': ['date', 'amount', 'vendor']
                    },
                    'metadata': {
                        'processing_time': 2.5,
                        'confidence_score': 0.95
                    }
                },
                response_only=True,
                status_codes=['200']
            ),
            OpenApiExample(
                'Validation Error',
                summary='File validation error',
                description='Error when no file is provided or unsupported format',
                value={'error': 'No image file provided'},
                response_only=True,
                status_codes=['400']
            ),
            OpenApiExample(
                'Processing Error',
                summary='Internal processing error',
                description='Error during document analysis processing',
                value={'error': 'Document analysis request failed: Processing error'},
                response_only=True,
                status_codes=['500']
            )
        ],
        tags=['Document Analysis']
    )
    def post(self, request):
        """
        Analyze a document image and extract structured information.
        
        Args:
            request: HTTP request containing the image file
            
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
            supported_formats = ['.jpg', '.jpeg', '.png']
            
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