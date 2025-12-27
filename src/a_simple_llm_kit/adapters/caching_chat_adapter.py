from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature


class CachingChatAdapter(ChatAdapter):
    """
    ChatAdapter that hoists role_instruction into the system message
    for better prompt caching (e.g., Anthropic's prompt caching).

    Use this when you have large static context (like CSV data) that should
    be cached across multiple calls with varying inputs.

    Example:
        adapter = CachingChatAdapter(role_instruction=csv_data)
        with dspy.context(adapter=adapter):
            result = predictor(focus_area="executive_summary")
    """

    def __init__(self, role_instruction: str = "", callbacks=None):
        super().__init__(callbacks=callbacks)
        self._role_instruction = role_instruction

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Prepend role_instruction to system message for caching."""
        base_structure = super().format_field_structure(signature)

        if self._role_instruction:
            return f"{self._role_instruction}\n\n{base_structure}"
        return base_structure

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict,
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Exclude role_instruction from user message (it's in system)."""
        filtered_inputs = {k: v for k, v in inputs.items() if k != "role_instruction"}

        return super().format_user_message_content(
            signature=signature,
            inputs=filtered_inputs,
            prefix=prefix,
            suffix=suffix,
            main_request=main_request,
        )
