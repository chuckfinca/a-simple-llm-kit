import dspy


class ContextSignature(dspy.Signature):
    """
    Base Signature for any module that requires file context and system instructions.
    All business logic modules should inherit from this.
    """

    # System logic (Mapped automatically from TaskContext)
    role_instruction: str = dspy.InputField(
        desc="Role, constraints, and behavior instructions."
    )
    context_documents: list[dspy.experimental.Document] = dspy.InputField(
        desc="Reference materials."
    )
    chat_history: str = dspy.InputField(desc="Conversation context.", default="")
