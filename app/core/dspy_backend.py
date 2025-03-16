from typing import Any, Type
from app.core.output_processors import DefaultOutputProcessor
from app.core.protocols import ModelBackend
from app.core.model_interfaces import Signature, ModelOutput
from app.core import logging

class DSPyModelBackendWithProcessor(ModelBackend):
    """Enhanced DSPy model backend that uses output processors"""
    
    def __init__(
        self,
        model_manager,
        model_id: str,
        signature_class: Type[Signature],
        output_processor=None,
        program_manager=None,
    ):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class
        self.program_manager = program_manager
        # Use the provided processor or default
        self.output_processor = output_processor or DefaultOutputProcessor()
        
        # Track program info if using program manager
        self.program_metadata = None
        if program_manager:
            # Register program if not already registered
            program_id = self._ensure_program_registration(signature_class)
            if not program_id:
                raise ValueError(
                    f"Failed to register {signature_class.__name__} with program manager. "
                    "This is required for versioning."
                )
    
    def _ensure_program_registration(self, signature_class):
        """Ensure the signature class is registered with the program manager"""
        # This method should contain the same implementation as in your current DSPyModelBackend
        # Reuse the existing implementation
        if not self.program_manager:
            logging.warning("No program manager available for program registration")
            return None
            
        # First, check if it's already registered by looking for matching signature class name
        registered_programs = self.program_manager.registry.list_programs()
        for prog in registered_programs:
            prog_class = self.program_manager.registry.get_program(prog.id)
            if prog_class and prog_class.__name__ == signature_class.__name__:
                self.program_metadata = prog
                logging.info(f"Found existing program registration for {signature_class.__name__} with ID {prog.id}")
                return prog.id
        
        # If not found, register it
        try:
            # Try to get a more descriptive name
            program_name = signature_class.__name__
            
            # Generate a more user-friendly name if possible (CamelCase -> Spaced Words)
            import re
            if re.match(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', program_name):
                program_name = re.sub(r'([A-Z])', r' \1', program_name).strip()
            
            self.program_metadata = self.program_manager.register_program(
                program_class=signature_class,
                name=program_name,
                description=signature_class.__doc__ or f"DSPy signature for {signature_class.__name__}"
            )
            logging.info(f"Successfully registered {signature_class.__name__} with ID {self.program_metadata.id}")
            return self.program_metadata.id
        except Exception as e:
            logging.error(f"Failed to register program {signature_class.__name__}: {str(e)}", exc_info=True)
            return None
    
    def _determine_input_key(self, signature_class, input_data):
        """Determine the appropriate input key for the signature"""
        # This method should contain the same implementation as in your current DSPyModelBackend
        # Reuse the existing implementation
        import dspy
        import inspect
        input_fields = {}
        
        # Analyze input fields from the signature class
        annotations = getattr(signature_class, '__annotations__', {})
        for name, field in annotations.items():
            # Look for fields marked as dspy.InputField
            if hasattr(field, "__origin__") and field.__origin__ == dspy.InputField:
                input_fields[name] = field
        
        # If no input fields found or only one input field, use generic approach
        if len(input_fields) <= 1:
            # Common input types
            if signature_class.__name__ == 'ContactExtractor':
                return {"image": input_data}
            else:
                return {"input": input_data}
        
        # For complex inputs with multiple fields, we need content to be a dict already
        if isinstance(input_data, dict):
            return input_data
        else:
            logging.warning(
                f"Complex signature {signature_class.__name__} with multiple inputs "
                f"but received non-dict input. Using 'input' as default key."
            )
            return {"input": input_data}
    
    async def predict(self, input_data: Any) -> ModelOutput:
        """Execute the model prediction with the signature class"""
        # This method should follow the same structure as in your current DSPyModelBackend
        # But replace the signature.process_output call with output_processor.process
        import uuid
        import dspy
        import asyncio
        
        max_attempts = 3
        base_delay = 1  # seconds
        
        # Generate a trace ID for this prediction
        trace_id = str(uuid.uuid4())
        
        # Convert input to the right format for program tracking
        input_dict = self._determine_input_key(self.signature, input_data)
        
        for attempt in range(max_attempts):
            try:
                lm = self.model_manager.models.get(self.model_id)
                if not lm:
                    raise ValueError(f"Model {self.model_id} not found")
                
                # If we have a program manager, use it for execution with tracking
                if self.program_manager and self.program_metadata:
                    result, execution_info = await self.program_manager.execute_program(
                        program_id=self.program_metadata.id,
                        model_id=self.model_id,
                        input_data=input_dict,
                        trace_id=trace_id
                    )
                    
                    # Add execution info to the result metadata
                    result_metadata = getattr(result, 'metadata', {})
                    if not isinstance(result_metadata, dict):
                        result_metadata = {}
                    
                    result_metadata['execution_info'] = execution_info.model_dump()
                    
                    # Ensure the result has a metadata attribute
                    if not hasattr(result, 'metadata'):
                        setattr(result, 'metadata', result_metadata)
                    else:
                        result.metadata = result_metadata
                    
                    # Use the processor instead of calling process_output directly
                    return self.output_processor.process(result)
                else:
                    # Direct execution should not happen if versioning is required
                    logging.warning(
                        f"Executing {self.signature.__name__} without program tracking. "
                        "This will cause versioning metadata to be missing."
                    )
                    
                    # Standard execution without tracking
                    dspy.configure(lm=lm)
                    predictor = dspy.Predict(self.signature)
                    
                    raw_result = predictor(**input_dict)
                    # Use the processor instead of calling process_output directly
                    return self.output_processor.process(raw_result)
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(
                        f"Final attempt failed for model {self.model_id}: {str(e)}",
                        extra={"trace_id": trace_id}
                    )
                    raise
                delay = base_delay * (2 ** attempt)
                logging.warning(
                    f"API call failed: {str(e)}, "
                    f"retrying in {delay} seconds... "
                    f"(attempt {attempt + 1}/{max_attempts})",
                    extra={"trace_id": trace_id}
                )
                await asyncio.sleep(delay)


# Factory function for creating backends with processors
def create_dspy_backend(
    model_manager, 
    model_id: str, 
    signature_class: Type[Signature],
    output_processor=None,
    program_manager=None
) -> ModelBackend:
    """
    Create a DSPy model backend with an optional custom output processor
    
    Args:
        model_manager: Model manager instance
        model_id: Model identifier
        signature_class: DSPy signature class
        output_processor: Optional custom output processor
        program_manager: Optional program manager for tracking
        
    Returns:
        Configured DSPyModelBackendWithProcessor instance
    """
    return DSPyModelBackendWithProcessor(
        model_manager,
        model_id,
        signature_class,
        output_processor,
        program_manager
    )