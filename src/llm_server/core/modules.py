from enum import Enum
from typing import Any, Generic, Optional, TypeVar

import dspy
import pydantic

# --- CONTEXT-SPECIFIC LABEL ENUMS ---


class PhoneFieldLabel(str, Enum):
    MOBILE = "mobile"
    WORK = "work"
    HOME = "home"
    MAIN = "main"
    PAGER = "pager"
    OTHER = "other"


class EmailFieldLabel(str, Enum):
    WORK = "work"
    HOME = "home"
    OTHER = "other"


class UrlFieldLabel(str, Enum):
    HOMEPAGE = "homepage"
    WORK = "work"
    HOME = "home"
    OTHER = "other"


class AddressFieldLabel(str, Enum):
    WORK = "work"
    HOME = "home"
    OTHER = "other"


# --- GENERIC LABELED VALUE MODEL ---
# A generic Pydantic model to handle any kind of labeled value.
T = TypeVar("T")
LabelT = TypeVar("LabelT")


class LabeledValue(pydantic.BaseModel, Generic[LabelT, T]):
    label: LabelT
    value: T


# --- CORE DATA MODELS ---


class PersonName(pydantic.BaseModel):
    prefix: Optional[str] = pydantic.Field(
        None, description="Title/prefix like Dr., Mr., Ms."
    )
    given_name: Optional[str] = pydantic.Field(None, description="First name")
    middle_name: Optional[str] = pydantic.Field(None, description="Middle name")
    family_name: Optional[str] = pydantic.Field(None, description="Last name")
    suffix: Optional[str] = pydantic.Field(None, description="Suffix like Jr., Ph.D.")


class WorkInformation(pydantic.BaseModel):
    job_title: Optional[str] = pydantic.Field(None, description="Professional title")
    department: Optional[str] = pydantic.Field(
        None, description="Department within organization"
    )
    organization_name: Optional[str] = pydantic.Field(
        None, description="Company or organization name"
    )


class PostalAddress(pydantic.BaseModel):
    street: Optional[str] = pydantic.Field(None, description="Street name and number")
    city: Optional[str] = pydantic.Field(None, description="City name")
    state: Optional[str] = pydantic.Field(None, description="State or province")
    postal_code: Optional[str] = pydantic.Field(None, description="ZIP or postal code")
    country: Optional[str] = pydantic.Field(None, description="Country name")


class SocialProfile(pydantic.BaseModel):
    service: Optional[str] = pydantic.Field(
        None, description="Name of social media service, e.g., 'twitter', 'linkedIn'"
    )
    username: Optional[str] = pydantic.Field(
        None, description="User handle or username on the service"
    )
    url_string: Optional[str] = pydantic.Field(
        None, description="The full URL to the user's profile."
    )


class ContactInformation(pydantic.BaseModel):
    # Each list now uses the generic LabeledValue with its specific Label enum.
    phone_numbers: list[LabeledValue[PhoneFieldLabel, str]] = pydantic.Field(
        default_factory=list
    )
    email_addresses: list[LabeledValue[EmailFieldLabel, str]] = pydantic.Field(
        default_factory=list
    )
    postal_addresses: list[LabeledValue[AddressFieldLabel, PostalAddress]] = (
        pydantic.Field(default_factory=list)
    )
    url_addresses: list[LabeledValue[UrlFieldLabel, str]] = pydantic.Field(
        default_factory=list
    )
    social_profiles: list[SocialProfile] = pydantic.Field(default_factory=list)


# --- TOP-LEVEL MODELS ---


class ExtractContact(pydantic.BaseModel):
    name: PersonName = pydantic.Field(default_factory=lambda: PersonName())  # pyright: ignore[reportCallIssue]
    work: WorkInformation = pydantic.Field(default_factory=lambda: WorkInformation())  # pyright: ignore[reportCallIssue]
    contact: ContactInformation = pydantic.Field(
        default_factory=lambda: ContactInformation()
    )

    notes: Optional[str] = pydantic.Field(
        None, description="Additional notes or information"
    )

    metadata: dict[str, Any] = {}
    model_config = pydantic.ConfigDict(extra="allow")

    def to_response(self) -> Any:
        return self


class ContactExtractor(dspy.Signature):
    """Extract contact information from an image."""

    image: dspy.Image = dspy.InputField()

    name: PersonName = dspy.OutputField()
    work: WorkInformation = dspy.OutputField()
    contact: ContactInformation = dspy.OutputField()
    notes: Optional[str] = dspy.OutputField()

    @classmethod
    def process_output(cls, result: Any) -> ExtractContact:
        """This method is now handled by the robust ContactExtractorProcessor and is effectively unused."""
        # This is a fallback and should ideally not be called.
        # The real processing happens in ContactExtractorProcessor.
        data_to_pass = {}
        if hasattr(result, "name"):
            data_to_pass["name"] = result.name
        if hasattr(result, "work"):
            data_to_pass["work"] = result.work
        if hasattr(result, "contact"):
            data_to_pass["contact"] = result.contact
        if hasattr(result, "notes"):
            data_to_pass["notes"] = result.notes
        return ExtractContact(**data_to_pass)

class RawContactExtractor(dspy.Signature):
    """
    A fallback signature that instructs the model to return all extracted contact
    information as a single, raw text block, ready for manual parsing.
    """

    image: dspy.Image = dspy.InputField()
    raw_data: str = dspy.OutputField()
