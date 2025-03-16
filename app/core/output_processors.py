from typing import Any
from app.core.model_interfaces import ModelOutput
from app.core.modules import ExtractContact, PersonName

class DefaultOutputProcessor:
    """Default processor for handling DSPy signature outputs"""
    
    def process(self, result: Any) -> ModelOutput:
        """
        Process raw DSPy output into a standardized ModelOutput
        
        Args:
            result: Raw result from DSPy predictor
            
        Returns:
            An object conforming to the ModelOutput protocol
        """
        # Preserve metadata if it exists
        metadata = {}
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            metadata = result.metadata
            
        # Create a standardized output object
        class StandardModelOutput:
            def __init__(self, output_value, metadata_dict=None):
                self.output = output_value
                self.metadata = metadata_dict or {}
                
            def to_response(self):
                """Convert to API response format"""
                return {
                    "response": self.output,
                    "metadata": self.metadata
                }
        
        # Extract output value from result 
        # (most DSPy signatures will have an 'output' attribute)
        output_value = None
        if hasattr(result, 'output'):
            output_value = result.output
        else:
            # Fallback: try to find any attribute that might be the output
            for attr_name in dir(result):
                if not attr_name.startswith('_') and attr_name not in ('metadata', 'to_response'):
                    output_value = getattr(result, attr_name)
                    break
        
        return StandardModelOutput(output_value, metadata)
    
class ContactExtractorProcessor:
    """Specialized processor for ContactExtractor signature outputs."""
    
    def process(self, result: Any) -> ExtractContact:
        """Process raw ContactExtractor output into a structured contact model."""
        # Preserve metadata from the original result
        metadata = getattr(result, 'metadata', {})
        
        # Validate required fields are present
        required_fields = ['given_name', 'phone_numbers', 'email_addresses']
        missing = [field for field in required_fields if not hasattr(result, field)]
        
        if missing:
            raise ValueError(
                f"ContactExtractorProcessor: Missing required fields: {', '.join(missing)}. "
                f"The signature output format may have changed."
            )
            
        # Create a contact object with the extracted information
        contact = ExtractContact(
            name=PersonName(
                prefix=getattr(result, 'name_prefix', None),
                given_name=result.given_name,  # Will raise AttributeError if missing
                # Other fields...
            ),
            # Rest of implementation...
        )
        
        return contact