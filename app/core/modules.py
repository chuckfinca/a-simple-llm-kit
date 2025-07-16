from typing import Any, Dict, Optional, List
import pydantic
import dspy
from enum import Enum

# --- NEW: Canonical Label Enum ---
# This is the single source of truth for all labels, making it cross-platform.
class ContactFieldLabel(str, Enum):
    # Phone Labels
    PHONE_MAIN = "main"
    PHONE_MOBILE = "mobile"
    PHONE_WORK = "work"
    PHONE_HOME = "home"
    PHONE_PAGER = "pager"
    OTHER_PHONE = "other_phone" # Renamed for clarity
    
    # Email Labels
    EMAIL_WORK = "work"
    EMAIL_HOME = "home"
    OTHER_EMAIL = "other_email"
    
    # URL Labels
    URL_HOMEPAGE = "homepage"
    URL_WORK = "work"
    URL_HOME = "home"
    OTHER_URL = "other_url"
    
    # Address Labels
    ADDRESS_WORK = "work"
    ADDRESS_HOME = "home"
    OTHER_ADDRESS = "other_address"

# --- Generic LabeledValue Model ---
class LabeledValue(pydantic.BaseModel):
    label: ContactFieldLabel
    value: str

# --- LabeledPostalAddress Model ---
class LabeledPostalAddress(pydantic.BaseModel):
    label: ContactFieldLabel
    value: "PostalAddress" # Use forward reference for the Pydantic model

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
    phone_numbers: List[LabeledValue] = pydantic.Field(default_factory=list, description="List of labeled phone numbers")
    email_addresses: List[LabeledValue] = pydantic.Field(default_factory=list, description="List of labeled email addresses")
    postal_addresses: List[LabeledPostalAddress] = pydantic.Field(default_factory=list, description="List of labeled postal addresses")
    url_addresses: List[LabeledValue] = pydantic.Field(default_factory=list, description="List of labeled websites/URLs")
    social_profiles: List["SocialProfile"] = pydantic.Field(default_factory=list, description="List of social media profiles") # Renamed for clarity

class SocialProfile(pydantic.BaseModel):
    service: str = pydantic.Field(description="Name of social media service, e.g., 'twitter', 'linkedIn'")
    username: str = pydantic.Field(description="User handle or username on the service")

# The main domain model that the API will return.
class ExtractContact(pydantic.BaseModel):
    name: PersonName
    work: WorkInformation
    contact: ContactInformation
    notes: Optional[str] = pydantic.Field(None, description="Additional notes or information")
    
    model_config = pydantic.ConfigDict(extra="allow")

class ContactExtractor(dspy.Signature):
    """Extracts structured contact information from an image of a business card.
    For each phone, email, and URL, provide a label from the allowed list.
    Allowed labels for phones: 'main', 'mobile', 'work', 'home', 'pager', 'other_phone'.
    Allowed labels for emails: 'work', 'home', 'other_email'.
    Allowed labels for URLs: 'homepage', 'work', 'home', 'other_url'.
    Allowed labels for addresses: 'work', 'home', 'other_address'.
    """
    image: dspy.Image = dspy.InputField(desc="An image of a business card.")

    # Pydantic models are used as OutputFields. DSPy will attempt to generate JSON.
    name: PersonName = dspy.OutputField(desc="The person's full name.")
    work: WorkInformation = dspy.OutputField(desc="The person's work and company information.")
    
    # This is the biggest change. We now ask for a single 'contact' object.
    contact: ContactInformation = dspy.OutputField(desc="All contact details including labeled phones, emails, URLs, addresses, and social profiles.")
    
    notes: Optional[str] = dspy.OutputField(desc="Any other relevant information or notes from the card.")

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
                
        contact = ExtractContact(
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
        
        # Preserve metadata if it exists on the result
        if hasattr(result, 'metadata'):
            setattr(contact, 'metadata', result.metadata)
        
        return contact
    