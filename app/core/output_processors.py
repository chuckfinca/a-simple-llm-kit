from typing import Any
from app.core.model_interfaces import ModelOutput
from app.core.modules import ExtractContact, PersonName, WorkInformation, ContactInformation, PostalAddress, SocialProfiles

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
        
        # Handle the case where the result is already an ExtractContact instance
        if isinstance(result, ExtractContact):
            return result
            
        # Validate required fields are present
        required_fields = ['given_name', 'phone_numbers', 'email_addresses']
        missing = [field for field in required_fields if not hasattr(result, field)]
        
        if missing:
            raise ValueError(
                f"ContactExtractorProcessor: Missing required fields: {', '.join(missing)}. "
                f"The signature output format may have changed."
            )
        
        # Get postal addresses
        postal_addresses = []
        if hasattr(result, 'postal_addresses'):
            for addr in result.postal_addresses:
                if isinstance(addr, PostalAddress):
                    postal_addresses.append(addr)
                elif isinstance(addr, dict):
                    postal_addresses.append(PostalAddress(**addr))
                else:
                    # Try to convert to dict first
                    addr_dict = {}
                    for key in ['street', 'city', 'state', 'postal_code', 'country']:
                        if hasattr(addr, key):
                            addr_dict[key] = getattr(addr, key)
                    postal_addresses.append(PostalAddress(**addr_dict))
        
        # Get social profiles
        social_profiles = []
        if hasattr(result, 'social_profiles'):
            for profile in result.social_profiles:
                if isinstance(profile, SocialProfiles):
                    social_profiles.append(profile)
                elif isinstance(profile, dict):
                    social_profiles.append(SocialProfiles(**profile))
                else:
                    # Try to convert to dict
                    prof_dict = {}
                    for key in ['service', 'url', 'username']:
                        if hasattr(profile, key):
                            prof_dict[key] = getattr(profile, key)
                    social_profiles.append(SocialProfiles(**prof_dict))
        
        # Create a contact object with the extracted information
        contact = ExtractContact(
            name=PersonName(
                prefix=getattr(result, 'name_prefix', None),
                given_name=getattr(result, 'given_name', None),
                middle_name=getattr(result, 'middle_name', None),
                family_name=getattr(result, 'family_name', None),
                suffix=getattr(result, 'name_suffix', None)
            ),
            work=WorkInformation(
                job_title=getattr(result, 'job_title', None),
                department=getattr(result, 'department_name', None),
                organization_name=getattr(result, 'organization_name', None)
            ),
            contact=ContactInformation(
                phone_numbers=getattr(result, 'phone_numbers', []),
                email_addresses=getattr(result, 'email_addresses', []),
                postal_addresses=postal_addresses,
                url_addresses=getattr(result, 'url_addresses', [])
            ),
            social=social_profiles,
            notes=getattr(result, 'notes', None)
        )
        
        # Add metadata
        if metadata:
            contact.metadata = metadata
        
        return contact