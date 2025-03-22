from typing import Dict, Any, Optional, List, Type
from app.core.protocols import PipelineStep, ModelBackend
from app.core.pipeline import Pipeline
from app.core.types import MediaType
from app.core.dspy_backend import create_dspy_backend
from app.core.implementations import ModelProcessor, ImageProcessor
from app.core.metrics_wrappers import PerformanceMetrics, TrackingFactory
from app.core.model_interfaces import Signature
from app.models.predictor import Predictor
from app.core.modules import ContactExtractor
from app.core.output_processors import DefaultOutputProcessor, ContactExtractorProcessor

def create_metrics_enabled_text_processor(
    model_manager, 
    model_id: str, 
    program_manager=None, 
    metadata: Optional[Dict[str, Any]] = None,
    output_processor=None,
    metrics: Optional[PerformanceMetrics] = None
) -> PipelineStep:
    """
    Create a text processor with metrics tracking
    
    Args:
        model_manager: Model manager instance
        model_id: Model identifier
        program_manager: Optional program manager
        metadata: Optional additional metadata
        output_processor: Optional custom output processor
        metrics: Optional metrics collector instance
        
    Returns:
        A text processor with metrics tracking
    """
    # Create metrics collector if not provided
    metrics = metrics or PerformanceMetrics()
    metrics.set_model_info(model_id)
    
    # Use original factory to create backend
    backend = create_dspy_backend(
        model_manager, 
        model_id, 
        Predictor, 
        output_processor=output_processor or DefaultOutputProcessor(),
        program_manager=program_manager
    )
    
    # Wrap backend with tracking
    tracked_backend = TrackingFactory.track_backend(backend, metrics)
    
    # Create processor with tracked backend
    processor = ModelProcessor(
        backend=tracked_backend,
        accepted_types=[MediaType.TEXT],
        output_type=MediaType.TEXT,
        metadata=metadata or {}
    )
    
    # Even for simple text processors, create a simple pipeline
    # This ensures we have consistent metrics even for single-step operations
    tracked_processor = TrackingFactory.track_step(processor, metrics, "TextProcessor")
    
    # Create a pipeline with just this one step
    # This generates more comprehensive metrics
    pipeline = Pipeline([tracked_processor])
    
    # Wrap whole pipeline with tracking
    return TrackingFactory.track_pipeline(pipeline, metrics)

def create_metrics_enabled_extract_contact_processor(
    model_manager, 
    model_id: str, 
    program_manager=None, 
    metadata: Optional[Dict[str, Any]] = None,
    output_processor=None,
    metrics: Optional[PerformanceMetrics] = None
) -> Pipeline:
    """
    Create a contact extraction pipeline with metrics tracking
    
    Args:
        model_manager: Model manager instance
        model_id: Model identifier
        program_manager: Optional program manager
        metadata: Optional additional metadata
        output_processor: Optional custom output processor
        metrics: Optional metrics collector instance
        
    Returns:
        A pipeline with metrics tracking
    """
    # Create metrics collector if not provided
    metrics = metrics or PerformanceMetrics()
    metrics.set_model_info(model_id)
    
    # Create components as usual
    contact_processor = output_processor or ContactExtractorProcessor()
    
    backend = create_dspy_backend(
        model_manager, 
        model_id, 
        ContactExtractor, 
        output_processor=contact_processor,
        program_manager=program_manager
    )
    
    # Wrap backend with tracking
    tracked_backend = TrackingFactory.track_backend(backend, metrics)
    
    # Create image processor
    image_processor = ImageProcessor()
    
    # Wrap image processor with tracking
    tracked_image_processor = TrackingFactory.track_step(
        image_processor, metrics, "ImageProcessor"
    )
    
    # Create model processor
    model_processor = ModelProcessor(
        backend=tracked_backend,
        accepted_types=[MediaType.IMAGE],
        output_type=MediaType.TEXT,
        metadata=metadata or {}
    )
    
    # Wrap model processor with tracking
    tracked_model_processor = TrackingFactory.track_step(
        model_processor, metrics, "ModelProcessor"
    )
    
    # Create pipeline
    pipeline = Pipeline([
        tracked_image_processor,
        tracked_model_processor
    ])
    
    # Wrap entire pipeline with tracking
    return TrackingFactory.track_pipeline(pipeline, metrics)

def create_metrics_enabled_processor_for_signature(
    model_manager,
    model_id: str,
    signature_class: Type[Signature],
    accepted_types: List[MediaType],
    output_type: MediaType,
    program_manager=None,
    metadata: Optional[Dict[str, Any]] = None,
    output_processor=None,
    metrics: Optional[PerformanceMetrics] = None
) -> PipelineStep:
    """
    Generic factory function to create a metrics-enabled processor for any DSPy signature
    
    Args:
        model_manager: Model manager instance
        model_id: Model identifier
        signature_class: DSPy signature class
        accepted_types: List of accepted media types
        output_type: Output media type
        program_manager: Optional program manager
        metadata: Optional additional metadata
        output_processor: Optional custom output processor
        metrics: Optional metrics collector instance
        
    Returns:
        A processor with metrics tracking
    """
    # Create metrics collector if not provided
    metrics = metrics or PerformanceMetrics()
    metrics.set_model_info(model_id)
    
    # Create backend normally
    backend = create_dspy_backend(
        model_manager,
        model_id,
        signature_class,
        output_processor=output_processor or DefaultOutputProcessor(),
        program_manager=program_manager
    )
    
    # Wrap backend with tracking
    tracked_backend = TrackingFactory.track_backend(backend, metrics)
    
    # Create processor
    processor = ModelProcessor(
        backend=tracked_backend,
        accepted_types=accepted_types,
        output_type=output_type,
        metadata=metadata or {}
    )
    
    # Wrap processor with tracking
    return TrackingFactory.track_step(
        processor, 
        metrics, 
        f"{signature_class.__name__}Processor"
    )