from typing import Optional, List
from typing_extensions import TypedDict

import dspy

class BusinessCardExtractor(dspy.Signature):
    """Extract business card information from an image."""

    image: dspy.Image = dspy.InputField()
    
    # Name Information
    name_prefix: Optional[str] = dspy.OutputField() #(description="Title/prefix like Dr., Mr., Ms.")
    given_name: Optional[str] = dspy.OutputField() #(description="First name")
    middle_name: Optional[str] = dspy.OutputField() #(description="Middle name")
    family_name: Optional[str] = dspy.OutputField() #(description="Last name")
    name_suffix: Optional[str] = dspy.OutputField() #(description="Suffix like Jr., Ph.D.")
    
    # Work Information
    job_title: Optional[str] = dspy.OutputField() #(description="Professional title")
    department_name: Optional[str] = dspy.OutputField() #(description="Department within organization")
    organization_name: Optional[str] = dspy.OutputField() #(description="Company or organization name")
    
    # Contact Information
    phone_numbers: List[str] = dspy.OutputField() #(description="List of phone numbers")
    email_addresses: List[str] = dspy.OutputField() #(description="List of email addresses")
    
    class PostalAddress(TypedDict, total=False):
        """Structured format for a postal address"""
        street: Optional[str]                   # Street name and number
        subLocality: Optional[str]              # Neighborhood or district
        city: Optional[str]                     # City name
        subAdministrativeArea: Optional[str]    # County or region
        state: Optional[str]                    # State or province
        postalCode: Optional[str]               # ZIP or postal code
        country: Optional[str]                  # Country name
        isoCountryCode: Optional[str]           # ISO country code

    postal_addresses: List[PostalAddress] = dspy.OutputField() #(description="List of postal addresses")
    url_addresses: List[str] = dspy.OutputField() #(description="List of websites/URLs")
    
    # Optional Information
    social_profiles: List[str] = dspy.OutputField() #(description="List of social media profiles")
    notes: Optional[str] = dspy.OutputField() #(description="Additional notes or information")