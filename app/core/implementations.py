import asyncio
from typing import Any, List, Union, Tuple, Optional, Dict
import uuid
import dspy
from PIL import Image
import io
import base64
from pathlib import Path

from app.core.protocols import PipelineStep, ModelBackend
from app.core.types import MediaType, PipelineData
from app.core.model_interfaces import Signature, ModelOutput
from app.core import logging

class DSPyModelBackend(ModelBackend):
    """DSPy model backend implementation with tracking and versioning support"""
    def __init__(
        self,
        model_manager,
        model_id: str,
        signature_class: type[Signature],
        program_manager=None,
    ):
        self.model_manager = model_manager
        self.model_id = model_id
        self.signature = signature_class
        self.program_manager = program_manager
        
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
    
    def _ensure_program_registration(self, signature_class: type[Signature]) -> Optional[str]:
        """
        Ensure the signature class is registered with the program manager.
        Returns the program ID if successful, None otherwise.
        """
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
    
    def _determine_input_key(self, signature_class: type[Signature], input_data: Any) -> Dict[str, Any]:
        """
        Determine the appropriate input key based on the signature class and input data.
        This handles different input formats for different program types.
        """
        # Get signature fields to determine proper input key
        import inspect
        input_fields = {}
        
        # Analyze input fields from the signature class
        for name, field in inspect.get_annotations(signature_class).items():
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
                    
                    return self.signature.process_output(result)
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
                    return self.signature.process_output(raw_result)
                    
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

class ModelProcessor:
    """Standard processor for model-based operations with metadata tracking"""
    def __init__(
        self, 
        backend: ModelBackend, 
        accepted_types: list[MediaType], 
        output_type: MediaType,
        metadata: Dict[str, Any] = None
    ):
        self.backend = backend
        self._accepted_types = accepted_types
        self.output_type = output_type
        self.metadata = metadata or {}
    
    async def process(self, data: PipelineData) -> PipelineData:
        result = await self.backend.predict(data.content)
        
        # Combine metadata from multiple sources with priority:
        # 1. Model execution specific metadata (from result)
        # 2. Processor instance metadata 
        # 3. Input data metadata
        combined_metadata = {
            **data.metadata,
            **self.metadata
        }
        
        # Add execution info if available from the result
        result_metadata = {}
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            result_metadata = result.metadata
        
        # If the backend has program metadata, include it
        if hasattr(self.backend, 'program_metadata') and self.backend.program_metadata:
            combined_metadata["program_metadata"] = self.backend.program_metadata.model_dump()
            
        combined_metadata.update(result_metadata)
        combined_metadata["processed"] = True
        
        return PipelineData(
            media_type=self.output_type,
            content=result,
            metadata=combined_metadata
        )
    
    @property
    def accepted_media_types(self) -> list[MediaType]:
        return self._accepted_types

class ImageContent:
    """Wrapper class to handle different image formats and conversions"""
    def __init__(self, content: Union[str, bytes]):
        self._content = content
        self._bytes: Optional[bytes] = None
        self._pil_image: Optional[Image.Image] = None
        self._data_uri: Optional[str] = None

    @property
    def bytes(self) -> bytes:
        """Get image as bytes, converting if necessary"""
        if self._bytes is None:
            if isinstance(self._content, bytes):
                self._bytes = self._content
            elif isinstance(self._content, str):
                if self._content.startswith('data:'):
                    # Handle data URI
                    _, base64_data = self._content.split(',', 1)
                    self._bytes = base64.b64decode(base64_data)
                else:
                    # Handle file path or base64 string
                    try:
                        with open(self._content, 'rb') as f:
                            self._bytes = f.read()
                    except:
                        # Try as base64
                        self._bytes = base64.b64decode(self._content)
        return self._bytes

    @property
    def pil_image(self) -> Image.Image:
        """Get as PIL Image, converting if necessary"""
        if self._pil_image is None:
            self._pil_image = Image.open(io.BytesIO(self.bytes))
            if self._pil_image.mode != 'RGB':
                self._pil_image = self._pil_image.convert('RGB')
        return self._pil_image

    @property
    def data_uri(self) -> str:
        """Get as data URI, converting if necessary"""
        if self._data_uri is None:
            mime_type = self.detect_mime_type()
            base64_data = base64.b64encode(self.bytes).decode('utf-8')
            self._data_uri = f"data:{mime_type};base64,{base64_data}"
        return self._data_uri

    def detect_mime_type(self) -> str:
        """Detect MIME type from image bytes"""
        if self.bytes.startswith(b'\x89PNG\r\n'):
            return 'image/png'
        if self.bytes.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        return 'image/png'  # Default to PNG

class ImageProcessor:
    """Combined image processing step that handles validation, conversion, and preprocessing"""
    def __init__(self, max_size: Tuple[int, int] = (800, 800)):
        self.max_size = max_size
        self._accepted_types = [MediaType.IMAGE]

    @property
    def accepted_media_types(self) -> List[MediaType]:
        return self._accepted_types

    async def process(self, data: PipelineData) -> PipelineData:
        # Wrap content in ImageContent for unified handling
        image = ImageContent(data.content)
        
        # Get original size before any processing
        original_size = image.pil_image.size
        
        # Calculate resize ratio if needed
        ratio = min(self.max_size[0] / original_size[0], 
                    self.max_size[1] / original_size[1])
        
        processed_size = original_size
        if ratio < 1:  # Only resize if image is larger than max_size
            processed_size = (
                int(original_size[0] * ratio),
                int(original_size[1] * ratio)
            )
            # Resize the image
            processed_pil = image.pil_image.resize(
                processed_size, 
                Image.Resampling.LANCZOS
            )
        else:
            # If no resize needed, use original
            processed_pil = image.pil_image

        # Convert to dspy.Image before returning
        processed_dspy = dspy.Image.from_PIL(processed_pil)

        return PipelineData(
            media_type=MediaType.IMAGE,
            content=processed_dspy,  # Now returning dspy.Image instead of PIL Image
            metadata={
                **data.metadata,
                'processed': True,
                'mime_type': image.detect_mime_type(),
                'original_size': original_size,
                'processed_size': processed_size,
                'compression_ratio': ratio if ratio < 1 else 1.0
            }
        )
