from typing import Any, List
from app.core.model_interfaces import ModelOutput
from app.core.modules import (
    ExtractContact,
    PersonName,
    WorkInformation,
    ContactInformation,
    PostalAddress,
    SocialProfile,
    LabeledValue,
    LabeledPostalAddress,
    ContactFieldLabel
)
from app.core import logging

class DefaultOutputProcessor:
    """Default processor for handling basic DSPy signature outputs."""
    
    def process(self, result: Any) -> Any:
        """
        Extracts the primary output from a DSPy result.
        
        This default implementation assumes the main content is in an 'output'
        attribute of the result object, which is common for simple DSPy signatures.
        """
        if hasattr(result, 'output'):
            return result.output
        
        # Fallback for other potential output formats
        # You can add more complex logic here if needed for other signatures
        logging.warning("DefaultOutputProcessor: Result has no 'output' attribute. Returning the result object itself.")
        return result

class ContactExtractorProcessor:
    """
    Specialized, robust processor for ContactExtractor signature outputs.
    This class acts as a firewall, validating and sanitizing the raw output
    from the LLM before it's sent to the mobile client.
    """
    
    def process(self, result: Any) -> ExtractContact:
        """
        Process raw ContactExtractor output into a validated ExtractContact model.
        This involves safe extraction, validation, and default fallbacks.
        """
        metadata = getattr(result, 'metadata', {})
        
        # Safely extract top-level objects, providing empty defaults if missing.
        # This prevents crashes if the LLM omits an entire section.
        name_info = self._get_attribute_as_model(result, 'name', PersonName)
        work_info = self._get_attribute_as_model(result, 'work', WorkInformation)
        contact_data = self._get_attribute_as_model(result, 'contact', ContactInformation)
        notes_info = getattr(result, 'notes', None)

        # --- VALIDATE AND SANITIZE THE CONTACT BLOCK ---
        
        # Sanitize each list of labeled values
        contact_data.phone_numbers = self._sanitize_labeled_list(
            contact_data.phone_numbers, ContactFieldLabel.OTHER_PHONE
        )
        contact_data.email_addresses = self._sanitize_labeled_list(
            contact_data.email_addresses, ContactFieldLabel.OTHER_EMAIL
        )
        contact_data.url_addresses = self._sanitize_labeled_list(
            contact_data.url_addresses, ContactFieldLabel.OTHER_URL
        )
        
        # Sanitize addresses, which have a nested structure
        contact_data.postal_addresses = self._sanitize_address_list(
            contact_data.postal_addresses, ContactFieldLabel.OTHER_ADDRESS
        )
        
        # Sanitize social profiles
        contact_data.social_profiles = self._sanitize_social_profiles(
            contact_data.social_profiles
        )
        
        # Construct the final, clean contact object
        contact = ExtractContact(
            name=name_info,
            work=work_info,
            contact=contact_data,
            notes=notes_info
        )
        
        # Re-attach the metadata so it's available in the final API response
        setattr(contact, 'metadata', metadata)
        
        return contact

    def _get_attribute_as_model(self, source: Any, attr_name: str, model_class: Any) -> Any:
        """Safely gets an attribute and converts it to a Pydantic model if it's a dict."""
        raw_attr = getattr(source, attr_name, None)
        
        if raw_attr is None:
            # LLM omitted the attribute, return a default empty model instance
            return model_class()
            
        if isinstance(raw_attr, dict):
            # LLM returned a dict, try to parse it into the Pydantic model
            try:
                return model_class(**raw_attr)
            except Exception as e:
                logging.warning(
                    f"Validation failed for '{attr_name}': Could not parse dict into {model_class.__name__}. "
                    f"Error: {e}. Returning default."
                )
                return model_class()
        
        # It's already the correct model type
        return raw_attr

    def _sanitize_labeled_list(self, items: List[Any], default_label: ContactFieldLabel) -> List[LabeledValue]:
        """Validates a list of LabeledValue items."""
        if not isinstance(items, list):
            logging.warning(f"Sanitization warning: Expected a list for labeled items, but got {type(items)}. Returning empty list.")
            return []

        sanitized_list = []
        valid_labels = {e.value for e in ContactFieldLabel}

        for item in items:
            try:
                # Ensure item is a dict-like object that can be parsed
                if not hasattr(item, 'label') or not hasattr(item, 'value'):
                    logging.warning(f"Sanitization warning: Skipping malformed item in list: {item}")
                    continue

                label_str = item.label.value

                # Check if the label is a valid member of our enum
                if label_str not in valid_labels:
                    logging.warning(
                        f"Validation warning: Invalid label '{label_str}' received from LLM. "
                        f"Defaulting to '{default_label.value}'."
                    )
                    item.label = default_label
                
                # Ensure value is a non-empty string
                if not isinstance(item.value, str) or not item.value.strip():
                    logging.warning(f"Sanitization warning: Skipping item with empty value: {item}")
                    continue
                    
                sanitized_list.append(LabeledValue(label=item.label, value=item.value))

            except Exception as e:
                logging.warning(f"Sanitization error processing labeled item '{item}': {e}")
                continue
                
        return sanitized_list

    def _sanitize_address_list(self, items: List[Any], default_label: ContactFieldLabel) -> List[LabeledPostalAddress]:
        """Validates a list of LabeledPostalAddress items."""
        if not isinstance(items, list):
            logging.warning(f"Sanitization warning: Expected a list for addresses, but got {type(items)}. Returning empty list.")
            return []

        sanitized_list = []
        valid_labels = {e.value for e in ContactFieldLabel}

        for item in items:
            try:
                if not hasattr(item, 'label') or not hasattr(item, 'value'):
                    continue

                label_str = str(item.label)
                if label_str not in valid_labels:
                    item.label = default_label

                # The 'value' should be a PostalAddress object or a dict
                address_value = self._get_attribute_as_model(item, 'value', PostalAddress)
                
                # Ensure the address is not completely empty
                if not any(vars(address_value).values()):
                    logging.warning(f"Sanitization warning: Skipping completely empty address object.")
                    continue
                    
                sanitized_list.append(LabeledPostalAddress(label=item.label, value=address_value))

            except Exception as e:
                logging.warning(f"Sanitization error processing address item '{item}': {e}")
                continue

        return sanitized_list

    def _sanitize_social_profiles(self, items: List[Any]) -> List[SocialProfile]:
        """Validates a list of SocialProfile items."""
        if not isinstance(items, list):
            logging.warning(f"Sanitization warning: Expected a list for social profiles, but got {type(items)}. Returning empty list.")
            return []
            
        sanitized_list = []
        for item in items:
            try:
                profile = self._get_attribute_as_model(item, 'self', SocialProfile) # 'self' because item itself is the model
                if not profile.service or not profile.username:
                    logging.warning(f"Sanitization warning: Skipping social profile with missing service or username: {profile}")
                    continue
                sanitized_list.append(profile)
            except Exception as e:
                logging.warning(f"Sanitization error processing social profile item '{item}': {e}")
                continue
        return sanitized_list