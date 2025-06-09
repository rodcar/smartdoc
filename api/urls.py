from django.urls import path
from .views import DocumentAnalysisView

urlpatterns = [
    path('analyze/', DocumentAnalysisView.as_view(), name='document-analysis'),
    #path('ocr/', ImageTextExtractionView.as_view(), name='image-ocr'),
]