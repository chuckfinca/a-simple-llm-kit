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


class DefaultOutputProcessor:
    def process(self, result: Any) -> Any:
        if hasattr(result, "output"):
            return result.output
        logging.warning("DefaultOutputProcessor: Result has no 'output' attribute.")
        return result


class ContactExtractorProcessor:
    def _extract_json_from_text(self, text: str, field_name: str) -> dict[str, Any]:
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
        if json_str.lower() == "null":
            return {}
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(
                f"Manual Parsing: JSON error for '{field_name}': {e}. Content: '{json_str}'"
            )
            return {}

    def process(self, result: Any) -> ExtractContact:
        metadata = getattr(result, "metadata", {})

        try:
            raw_output_text = getattr(result, "raw_completion", None)
            if not raw_output_text or not isinstance(raw_output_text, str):
                raise AttributeError(
                    "The 'raw_completion' attribute is missing from the result object."
                )
        except AttributeError as e:
            logging.error(f"CRITICAL PARSING FAILURE: {e}", exc_info=True)
            return self._process_with_fallback(result)

        name_data = self._extract_json_from_text(raw_output_text, "name")
        work_data = self._extract_json_from_text(raw_output_text, "work")
        contact_data = self._extract_json_from_text(raw_output_text, "contact")

        notes_match = re.search(
            r"\[\[ ## notes ## \]\]\s*```json\s*([\s\S]*?)\s*```",
            raw_output_text,
            re.DOTALL,
        )
        final_notes = None
        if notes_match:
            notes_str = notes_match.group(1).strip()
            if notes_str.lower() != "null":
                try:
                    final_notes = json.loads(notes_str)
                except json.JSONDecodeError:
                    final_notes = notes_str

        try:
            contact = ExtractContact(
                name=PersonName(**name_data),
                work=WorkInformation(**work_data),
                contact=ContactInformation(**contact_data),
                notes=final_notes if isinstance(final_notes, str) else None,
            )
            contact.metadata = metadata
            logging.info("Manual parsing successful. All contact fields processed.")
            return contact
        except Exception as e:
            logging.error(
                f"Pydantic validation failed after manual parsing. Error: {e}",
                exc_info=True,
            )
            return self._process_with_fallback(result)

    def _process_with_fallback(self, result: Any) -> ExtractContact:
        logging.warning("Executing fallback due to critical parsing error.")

        data_to_pass = {}
        if hasattr(result, "name"):
            data_to_pass["name"] = result.name
        if hasattr(result, "work"):
            data_to_pass["work"] = result.work
        if hasattr(result, "contact"):
            data_to_pass["contact"] = result.contact
        if hasattr(result, "notes"):
            data_to_pass["notes"] = result.notes
        contact = ExtractContact(**data_to_pass)
        contact.metadata = getattr(result, "metadata", {})
        # setattr(contact, "metadata", getattr(result, "metadata", {}))
        return contact
