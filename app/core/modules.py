from typing import Any, Optional, List
import pydantic
import dspy

class PersonName(pydantic.BaseModel):
    """Structured format for a person's name"""
    prefix: Optional[str] = pydantic.Field(None, description="Title/prefix like Dr., Mr., Ms.")
    given_name: Optional[str] = pydantic.Field(None, description="First name")
    middle_name: Optional[str] = pydantic.Field(None, description="Middle name")
    family_name: Optional[str] = pydantic.Field(None, description="Last name")
    suffix: Optional[str] = pydantic.Field(None, description="Suffix like Jr., Ph.D.")

class WorkInformation(pydantic.BaseModel):
    """Structured format for work-related information"""
    job_title: Optional[str] = pydantic.Field(None, description="Professional title")
    department: Optional[str] = pydantic.Field(None, description="Department within organization")
    organization_name: Optional[str] = pydantic.Field(None, description="Company or organization name")

class PostalAddress(pydantic.BaseModel):
    """Structured format for a postal address"""
    street: Optional[str] = pydantic.Field(None, description="Street name and number")
    sub_locality: Optional[str] = pydantic.Field(None, description="Neighborhood or district")
    city: Optional[str] = pydantic.Field(None, description="City name")
    sub_administrative_area: Optional[str] = pydantic.Field(None, description="County or region")
    state: Optional[str] = pydantic.Field(None, description="State or province")
    postal_code: Optional[str] = pydantic.Field(None, description="ZIP or postal code")
    country: Optional[str] = pydantic.Field(None, description="Country name")
    iso_country_code: Optional[str] = pydantic.Field(None, description="ISO country code")

class ContactInformation(pydantic.BaseModel):
    """Structured format for contact information"""
    phone_numbers: List[str] = pydantic.Field(default_factory=list, description="List of phone numbers")
    email_addresses: List[str] = pydantic.Field(default_factory=list, description="List of email addresses")
    postal_addresses: List[PostalAddress] = pydantic.Field(default_factory=list, description="List of postal addresses")
    url_addresses: List[str] = pydantic.Field(default_factory=list, description="List of websites/URLs")
    social_profiles: List[str] = pydantic.Field(default_factory=list, description="List of social media profiles")

class SocialProfiles(pydantic.BaseModel):
    """Structured format for social profile information"""
    service: str = pydantic.Field(default_factory=list, description="Name of social media service")
    url: Optional[str] = pydantic.Field(default_factory=list, description="URL of service")
    username: str = pydantic.Field(default_factory=list, description="User handle")

class ExtractContact(pydantic.BaseModel):
    """Domain model for contact data"""
    name: PersonName
    work: WorkInformation
    contact: ContactInformation
    social: List[SocialProfiles]
    notes: Optional[str] = pydantic.Field(None, description="Additional notes or information")

class ContactExtractor(dspy.Signature):
    """Extract contact information from an image."""
    image: dspy.Image = dspy.InputField()
    
    # Name Information
    name_prefix: Optional[str] = dspy.OutputField()
    given_name: Optional[str] = dspy.OutputField()
    middle_name: Optional[str] = dspy.OutputField()
    family_name: Optional[str] = dspy.OutputField()
    name_suffix: Optional[str] = dspy.OutputField()
    
    # Work Information
    job_title: Optional[str] = dspy.OutputField()
    department_name: Optional[str] = dspy.OutputField()
    organization_name: Optional[str] = dspy.OutputField()
    
    # Contact Information
    phone_numbers: List[str] = dspy.OutputField()
    email_addresses: List[str] = dspy.OutputField()
    postal_addresses: List[PostalAddress] = dspy.OutputField()
    url_addresses: List[str] = dspy.OutputField()
    social_profiles: List[SocialProfiles] = dspy.OutputField()
    notes: Optional[str] = dspy.OutputField()

    @classmethod
    def process_output(cls, result: Any) -> ExtractContact:
        """Process raw output into validated ExtractContact domain model"""
        
        # Handle the case where addresses might already be PostalAddress objects
        postal_addresses = []
        for addr in result.postal_addresses:
            if isinstance(addr, PostalAddress):
                postal_addresses.append(addr)
            else:
                # It's a dictionary, so unpack it
                postal_addresses.append(PostalAddress(**addr))
                
        return ExtractContact(
            name=PersonName(
                prefix=result.name_prefix,
                given_name=result.given_name,
                middle_name=result.middle_name,
                family_name=result.family_name,
                suffix=result.name_suffix
            ),
            work=WorkInformation(
                job_title=result.job_title,
                department=result.department_name,
                organization_name=result.organization_name
            ),
            contact=ContactInformation(
                phone_numbers=result.phone_numbers,
                email_addresses=result.email_addresses,
                postal_addresses=postal_addresses,
                url_addresses=result.url_addresses
            ),
            social=[SocialProfiles(**social) for social in result.social_profiles],
            notes=result.notes
        )
