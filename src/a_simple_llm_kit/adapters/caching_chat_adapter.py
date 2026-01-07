from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature


class CachingChatAdapter(ChatAdapter):
    """
    ChatAdapter that prepends system instructions to the system message
    for better prompt caching (e.g., Anthropic's prompt caching).

    Example:
        adapter = CachingChatAdapter(system_instructions="Always be concise.")
        with dspy.context(adapter=adapter):
            result = predictor(tab_name="Executive Summary")
    """

    def __init__(self, system_instructions: str = "", callbacks=None):
        super().__init__(callbacks=callbacks)
        self._system_instructions = system_instructions

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Prepend system_instructions to system message."""
        base_structure = super().format_field_structure(signature)

        if self._system_instructions:
            return f"{self._system_instructions}\n\n{base_structure}"
        return base_structure
