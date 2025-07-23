import json
import re
from typing import Any
from llm_server.core import logging
from llm_server.core.modules import (
    ContactInformation,
    ExtractContact,
    PersonName,
    WorkInformation,
)
import dspy

class DefaultOutputProcessor:
    def process(self, result: Any) -> Any:
        if hasattr(result, "output"):
            return result.output
        # Also handle cases where the result might be from a TypedPredictor
        if hasattr(result, "model_dump_json"):
             try:
                 # It's a Pydantic model
                 return json.loads(result.model_dump_json())
             except Exception:
                 pass # Fallback to returning the object
        logging.warning("DefaultOutputProcessor: Result has no 'output' attribute and is not a standard Pydantic model.")
        return result

class ContactExtractorProcessor:
    """
    Processes output for contact extraction. It handles both structured Pydantic
    objects from TypedPredictor and raw text from a fallback mechanism.
    """

    def process(self, result: Any) -> ExtractContact:
        """
        Processes the result from a DSPy predictor.

        If the result is already a structured ExtractContact object (the happy path),
        it returns it directly.

        If the result contains raw text (the fallback path), it uses robust
        manual parsing to construct the ExtractContact object.
        """
        # Happy Path: Result is already a structured Pydantic object.
        # DSPy's TypedPredictor returns a dspy.Prediction object, and the first field is the Pydantic model.
        if hasattr(result, 'name') and isinstance(result.name, PersonName):
            logging.info("Successfully processed structured output from TypedPredictor.")
            # The result itself has the correct structure, so we can build the object.
            return ExtractContact(
                name=result.name,
                work=result.work,
                contact=result.contact,
                notes=result.notes,
                metadata=getattr(result, "metadata", {})
            )

        # Fallback Path: Result is a prediction with raw text that needs parsing.
        elif hasattr(result, "raw_contact_data"):
            logging.warning("Falling back to manual parsing for raw text output.")
            raw_output_text = result.raw_contact_data
            return self._parse_raw_text(raw_output_text, getattr(result, "metadata", {}))

        # If we receive an unexpected format
        else:
            logging.error(f"CRITICAL PARSING FAILURE: Received unexpected result format: {type(result)}", exc_info=True)
            # Create an empty contact to avoid a crash, but log the failure.
            return ExtractContact(metadata={"parsing_error": "Received unexpected result format"})


    def _parse_raw_text(self, raw_output_text: str, metadata: dict) -> ExtractContact:
        """The robust manual parsing logic you already wrote."""
        name_data = self._extract_json_from_text(raw_output_text, "name")
        work_data = self._extract_json_from_text(raw_output_text, "work")
        contact_data = self._extract_json_from_text(raw_output_text, "contact")
        notes_data = self._extract_json_from_text(raw_output_text, "notes")

        final_notes = notes_data if isinstance(notes_data, str) else None

        try:
            contact = ExtractContact(
                name=PersonName(**name_data),
                work=WorkInformation(**work_data),
                contact=ContactInformation(**contact_data),
                notes=final_notes,
                metadata=metadata
            )
            logging.info("Manual parsing successful. All contact fields processed.")
            return contact
        except Exception as e:
            logging.error(f"Pydantic validation failed after manual parsing. Error: {e}", exc_info=True)
            # Return a partially filled object on validation failure
            return ExtractContact(metadata={"parsing_error": str(e), **metadata})


    def _extract_json_from_text(self, text: str, field_name: str) -> dict[str, Any]:
        """Your existing, proven JSON extraction logic."""
        pattern = re.compile(
            r"\[\[ ## "
            + re.escape(field_name)
            + r" ## \]\]\s*```json\s*([\s\S]*?)\s*```",
            re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            logging.warning(f"Manual Parsing: No marker for '{field_name}'.")
            return {}
        json_str = match.group(1).strip()
        if json_str.lower() in ["null", "none"]:
            return {}
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Manual Parsing: JSON error for '{field_name}': {e}. Content: '{json_str}'")
            return {}
            