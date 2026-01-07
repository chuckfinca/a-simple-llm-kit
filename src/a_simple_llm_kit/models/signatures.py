import dspy


class ContextSignature(dspy.Signature):
    """
    Base Signature for modules that require document context.
    Task description should be in the child class docstring.
    All business logic modules should inherit from this.
    """

    # Context data (budget tables, metrics, PDF content, etc.)
    context_documents: str = dspy.InputField(
        desc="Reference data and documents for analysis."
    )
    chat_history: str = dspy.InputField(desc="Conversation context.", default="")
