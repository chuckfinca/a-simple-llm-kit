import uuid
import dspy
import asyncio
from typing import Any, Type
from app.core.output_processors import DefaultOutputProcessor
from app.core.protocols import ModelBackend
from app.core.model_interfaces import Signature, ModelOutput
from app.core import logging
from app.core.utils import ensure_program_metadata_object

class DSPyModelBackendWithProcessor(ModelBackend):
    """Enhanced DSPy model backend that receives raw text for robust processing."""
    
    def __init__(
        self, model_manager, model_id: str, signature_class: Type[Signature],
        output_processor=None, program_manager=None,
    ):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class
        self.program_manager = program_manager
        self.output_processor = output_processor or DefaultOutputProcessor()
        self.program_metadata = None
        if program_manager:
            if not self._ensure_program_registration(signature_class):
                raise ValueError(f"Failed to register {signature_class.__name__} with program manager.")

    def _ensure_program_registration(self, signature_class):
        if not self.program_manager: return None
        for prog in self.program_manager.registry.list_programs():
            prog_class = self.program_manager.registry.get_program(prog.id)
            if prog_class and prog_class.__name__ == signature_class.__name__:
                self.program_metadata = prog
                return prog.id
        try:
            import re
            program_name = signature_class.__name__
            if re.match(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', program_name):
                program_name = re.sub(r'([A-Z])', r' \1', program_name).strip()
            self.program_metadata = self.program_manager.register_program(
                program_class=signature_class, name=program_name,
                description=signature_class.__doc__ or f"DSPy signature for {signature_class.__name__}"
            )
            return self.program_metadata.id
        except Exception as e:
            logging.error(f"Failed to register program {signature_class.__name__}: {str(e)}", exc_info=True)
            return None
    
    def _determine_input_key(self, signature_class, input_data):
        import dspy
        annotations = getattr(signature_class, '__annotations__', {})
        is_contact_extractor = signature_class.__name__ == 'ContactExtractor'
        return {"image": input_data} if is_contact_extractor else {"input": input_data}
    
    async def predict(self, input_data: Any) -> ModelOutput:
        max_attempts, base_delay = 3, 1
        trace_id = str(uuid.uuid4())
        input_dict = self._determine_input_key(self.signature, input_data)
        
        for attempt in range(max_attempts):
            try:
                lm = self.model_manager.models.get(self.model_id)
                if not lm: raise ValueError(f"Model {self.model_id} not found")

                if self.program_manager and self.program_metadata:
                    result, execution_info, raw_completion_text = await self.program_manager.execute_program(
                        program_id=self.program_metadata.id, model_id=self.model_id,
                        input_data=input_dict, trace_id=trace_id
                    )
                    
                    # Attach the raw text to the result object for the processor
                    setattr(result, 'raw_completion', raw_completion_text)
                    
                    metadata = getattr(result, 'metadata', {})
                    metadata['execution_info'] = execution_info.model_dump()
                    metadata['program_metadata'] = ensure_program_metadata_object(self.program_metadata)
                    result.metadata = metadata
                    
                    return self.output_processor.process(result)
                else:
                    dspy.configure(lm=lm)
                    predictor = dspy.Predict(self.signature)
                    result = predictor(**input_dict)
                    return self.output_processor.process(result)
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    logging.error(f"Final attempt failed for model {self.model_id}: {str(e)}", extra={"trace_id": trace_id})
                    raise
                delay = base_delay * (2 ** attempt)
                logging.warning(f"API call failed: {str(e)}, retrying in {delay}s...", extra={"trace_id": trace_id})
                await asyncio.sleep(delay)
    
    def get_lm_history(self):
        try:
            return self.model_manager.models.get(self.model_id).history
        except Exception as e:
            logging.warning(f"Could not access LM history: {str(e)}")
        return []

def create_dspy_backend(model_manager, model_id: str, signature_class: Type[Signature], output_processor=None, program_manager=None) -> ModelBackend:
    return DSPyModelBackendWithProcessor(model_manager, model_id, signature_class, output_processor, program_manager)