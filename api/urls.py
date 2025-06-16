from django.urls import path
from .views import DocumentAnalysisView

urlpatterns = [
    path('analyze/', DocumentAnalysisView.as_view(), name='document-analysis')
]