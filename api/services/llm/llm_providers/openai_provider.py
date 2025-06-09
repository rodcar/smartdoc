# api/services/llm/llm_providers/openai_provider.py
import os
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

from .base import LLMProvider


class DocumentClassificationResponse(BaseModel):
    predicted_category: str = Field(..., description="The predicted document category")


class DocumentEntityExtractionResponse(BaseModel):
    entity1_name: Optional[str] = Field(None, description="Name of first entity")
    entity1_value: Optional[str] = Field(None, description="Value of first entity")
    entity1_description: Optional[str] = Field(None, description="Description of first entity")
    entity2_name: Optional[str] = Field(None, description="Name of second entity")
    entity2_value: Optional[str] = Field(None, description="Value of second entity")
    entity2_description: Optional[str] = Field(None, description="Description of second entity")
    entity3_name: Optional[str] = Field(None, description="Name of third entity")
    entity3_value: Optional[str] = Field(None, description="Value of third entity")
    entity3_description: Optional[str] = Field(None, description="Description of third entity")
    document_type: str = Field(..., description="The document type")


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation using GPT models.
    """
    
    def __init__(self, model: str = "gpt-4.1", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    def classify_document(self, 
                         image_base64: str, 
                         extracted_text: str, 
                         categories: List[str],
                         confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Classify a document using OpenAI's vision capabilities.
        
        Args:
            image_base64: Base64 encoded image
            extracted_text: Text extracted from the image via OCR
            categories: List of possible document categories
            confidence_threshold: Minimum confidence threshold for classification
            
        Returns:
            dict: Classification result
        """
        try:
            # Create the classification prompt
            classification_prompt = f"""
            You are a document classification expert. Analyze the provided image and extracted text to classify the document into one of these specific categories:
            
            Categories: {', '.join(categories)}
            
            Extracted Text:
            {extracted_text if extracted_text else 'No text extracted'}...
            
            Based on the visual elements in the image and the extracted text content, classify this document into exactly ONE of the predefined categories.
            
            Return only the predicted_category field with the exact category name from the list.
            """
            
            # Use OpenAI Responses API for structured classification
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": classification_prompt },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        ],
                    }
                ],
                text_format=DocumentClassificationResponse
            )
            
            # Get the parsed response
            classification_result = response.output_parsed
            
            # Validate the prediction is in our categories
            if classification_result.predicted_category.lower() in [cat.lower() for cat in categories]:
                # Find the original case category
                predicted_category = next(cat for cat in categories if cat.lower() == classification_result.predicted_category.lower())
                
                return {
                    'predicted_category': predicted_category,
                    'success': True,
                    'error': None
                }
            else:
                return {
                    'predicted_category': 'unknown',
                    'success': False,
                    'error': 'Classification result not in valid categories'
                }
                
        except Exception as e:
            return {
                'predicted_category': 'error',
                'success': False,
                'error': f"OpenAI classification failed: {str(e)}"
            }

    def extract_entities(self,
                        image_base64: str,
                        extracted_text: str,
                        document_type: str,
                        confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Extract relevant entities from a document based on its type using OpenAI's vision capabilities.
        
        Args:
            image_base64: Base64 encoded image
            extracted_text: Text extracted from the image via OCR
            document_type: The predicted document type to guide entity extraction
            confidence_threshold: Minimum confidence threshold for entity extraction
            
        Returns:
            dict: Entity extraction result
        """
        try:
            # Create the entity extraction prompt
            entity_prompt = f"""
            You are an expert document entity extraction system. Analyze the provided image and extracted text to identify and extract relevant entities based on the document type.

            Document Type: {document_type}
            
            Extracted Text:
            {extracted_text if extracted_text else 'No text extracted'}...
            
            Instructions:
            1. Examine both the image and the extracted text
            2. Extract up to 3 most important entities that are relevant for a {document_type} document
            3. Use descriptive names for entities. For example: 'invoice_number', 'vendor_name', 'total_amount', etc.
            4. For each entity, provide the extracted value and a brief description
            5. Include the document type in the response
            
            Extract all relevant entities and provide structured output.
            """
            
            # Use OpenAI Responses API for structured entity extraction
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": entity_prompt },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        ],
                    }
                ],
                text_format=DocumentEntityExtractionResponse
            )
            
            # Get the parsed response
            entity_result = response.output_parsed
            
            # Convert the structured response to the expected format
            entities_list = []
            
            # Process the flat entity structure
            for i in range(1, 4):  # Support up to 3 entities
                entity_name = getattr(entity_result, f'entity{i}_name', None)
                entity_value = getattr(entity_result, f'entity{i}_value', None)
                
                if entity_name and entity_value:
                    entities_list.append({
                        "name": entity_name,
                        "value": entity_value
                    })
            
            return {
                'entities': entities_list,
                'document_type': document_type,
                'success': True,
                'error': None
            }
                
        except Exception as e:
            return {
                'entities': [],
                'document_type': document_type,
                'success': False,
                'error': f"OpenAI entity extraction failed: {str(e)}"
            }
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's text generation capabilities.
        
        Args:
            prompt: Text prompt to generate from
            **kwargs: Additional parameters for OpenAI API
            
        Returns:
            str: Generated text response
        """
        try:
            # Default parameters
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
            }
            
            # Override with any provided kwargs
            params.update(kwargs)
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI text generation failed: {str(e)}")
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return f"OpenAI ({self.model})"
    
    @property
    def supported_features(self) -> List[str]:
        """Return supported features."""
        return ["classification", "vision", "text_generation", "entity_extraction"]
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
        try:
            return self.api_key is not None and len(self.api_key.strip()) > 0
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "supports_vision": True,
            "supports_classification": True,
            "supports_text_generation": True,
            "supports_entity_extraction": True
        } 