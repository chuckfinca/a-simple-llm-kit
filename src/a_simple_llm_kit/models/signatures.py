import dspy
from dspy.experimental import Document


class ContextSignature(dspy.Signature):
    """
    Base Signature for any module that requires file context and system instructions.
    All business logic modules should inherit from this.
    """

    # System logic (Mapped automatically from TaskContext)
    role_instruction: str = dspy.InputField(
        desc="Role, constraints, and behavior instructions."
    )
    context_documents: list[Document] = dspy.InputField(
        desc="Reference materials (text, PDFs, etc.)."
    )
    chat_history: str = dspy.InputField(desc="Conversation context.", default="")
