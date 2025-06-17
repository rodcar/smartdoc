# api/services/llm/llm_providers/openai_provider.py
import os
import json
import time
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field

from .base import LLMProvider


class DocumentClassificationResponse(BaseModel):
    document_type: str = Field(..., description="The predicted document category")


class Entity(BaseModel):
    value: str = Field(..., description="The actual value extracted")
    description: str = Field(..., description="The description of the entity extracted")


class DocumentEntityExtractionResponse(BaseModel):
    entities: List[Entity] = Field(default_factory=list, description="List of extracted entities")


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation using GPT models.
    """
    
    def __init__(self, model: str = "o3-2025-04-16", api_key: Optional[str] = None, timeout: float = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout or float(os.getenv("OPENAI_TIMEOUT", 120.0))  # Default 2 minutes
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, timeout=self.timeout)
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
            You are a document classification expert. Your task is to classify the provided document into EXACTLY ONE of the following predefined categories:

            Available Categories:
            {', '.join(categories)}
            
            Extracted Text from Document:
            {extracted_text if extracted_text else 'No text extracted'}
            
            Based on the extracted text content above, classify this document into exactly ONE of the predefined categories.
            
            Return only the document_type field with the exact category name from the list.
            """
            
            # Use OpenAI Responses API (2025) with text only (no image to avoid hanging)
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "input_text", "text": classification_prompt }
                        ],
                    }
                ],
                text_format=DocumentClassificationResponse
            )
            
            # Get the parsed response
            classification_result = response.output_parsed
            
            # Validate the prediction
            document_type = classification_result.document_type
            if document_type in categories:
                return {
                    'document_type': document_type,
                    'success': True,
                    'error': None
                }
            else:
                return {
                    'document_type': 'unknown',
                    'success': True,
                    'error': f'Unknown category: {document_type}'
                }
                
        except Exception as e:
            return {
                'document_type': 'error',
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
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Create the entity extraction prompt
                entity_prompt = f"""
                You are an expert document entity extraction system. Analyze the extracted text to identify and extract relevant entities based on the document type.

                Document Type: {document_type}
                
                Extracted Text from Document:
                {extracted_text if extracted_text else 'No text extracted'}
                
                Instructions:
                1. Examine the extracted text above
                2. Extract all relevant entities for this document type
                3. For each entity, provide the extracted value and a brief description
                
                Extract all entities and provide structured output.
                """
                
                # Use OpenAI Responses API (2025) with text only (no image to avoid hanging)
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                { "type": "input_text", "text": entity_prompt }
                            ],
                        }
                    ],
                    text_format=DocumentEntityExtractionResponse
                )
                
                # Get the parsed response
                entity_result = response.output_parsed
                
                # Convert the structured response to the expected format
                entities_list = []
                
                # Process the entity list from the response
                for entity in entity_result.entities:
                    entities_list.append({
                        "value": entity.value,
                        "description": entity.description
                    })
                
                return {
                    'entities': entities_list,
                    'success': True,
                    'error': None
                }
                    
            except Exception as e:
                error_msg = str(e)
                is_timeout = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                
                if attempt < max_retries - 1 and is_timeout:
                    # Exponential backoff for timeout errors
                    delay = base_delay * (2 ** attempt)
                    print(f"OpenAI timeout on attempt {attempt + 1}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                
                return {
                    'entities': [],
                    'success': False,
                    'error': f"OpenAI entity extraction failed: {error_msg}"
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