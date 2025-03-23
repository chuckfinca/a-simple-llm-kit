from typing import Dict, Any, Optional, List, Type, Tuple
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

# Helper function to setup metrics with model info (unchanged)
def _setup_metrics_with_model_info(metrics, model_id, program_manager):
    """Helper to initialize metrics with proper model information"""
    metrics = metrics or PerformanceMetrics()
    if program_manager and hasattr(program_manager, 'model_info') and model_id in program_manager.model_info:
        metrics.set_model_info(model_id, program_manager.model_info[model_id])
    else:
        metrics.set_model_info(model_id)
    return metrics

def _create_metrics_enabled_base(
    model_manager, 
    model_id: str,
    signature_class: Type[Signature],
    program_manager=None,
    metadata: Optional[Dict[str, Any]] = None,
    output_processor=None,
    metrics: Optional[PerformanceMetrics] = None
) -> Tuple[ModelBackend, PerformanceMetrics]:
    """
    Base factory method for creating metrics-enabled components
    
    Args:
        model_manager: Model manager instance
        model_id: Model identifier
        signature_class: DSPy signature class
        program_manager: Optional program manager
        metadata: Optional additional metadata
        output_processor: Optional custom output processor
        metrics: Optional metrics collector instance
        
    Returns:
        Tuple of (tracked backend, metrics instance)
    """
    # Initialize metrics with proper model information
    metrics = _setup_metrics_with_model_info(metrics, model_id, program_manager)
    
    # Create backend
    backend = create_dspy_backend(
        model_manager, 
        model_id, 
        signature_class, 
        output_processor=output_processor,
        program_manager=program_manager
    )
    
    # Wrap backend with tracking
    tracked_backend = TrackingFactory.track_backend(backend, metrics)
    
    return tracked_backend, metrics

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
    # Use base factory to get tracked backend and metrics
    tracked_backend, metrics = _create_metrics_enabled_base(
        model_manager, 
        model_id, 
        Predictor,
        program_manager, 
        metadata, 
        output_processor or DefaultOutputProcessor(),
        metrics
    )
    
    # Create processor with tracked backend
    processor = ModelProcessor(
        backend=tracked_backend,
        accepted_types=[MediaType.TEXT],
        output_type=MediaType.TEXT,
        metadata=metadata or {}
    )
    
    # Track the processor
    tracked_processor = TrackingFactory.track_step(processor, metrics, "TextProcessor")
    
    # Create a pipeline with the tracked processor
    pipeline = Pipeline([tracked_processor])
    
    # Return the tracked pipeline
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
    # Use base factory to get tracked backend and metrics
    tracked_backend, metrics = _create_metrics_enabled_base(
        model_manager, 
        model_id, 
        ContactExtractor,
        program_manager, 
        metadata, 
        output_processor or ContactExtractorProcessor(),
        metrics
    )
    
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
    
    # Wrap pipeline with tracking
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
    # Use base factory to get tracked backend and metrics
    tracked_backend, metrics = _create_metrics_enabled_base(
        model_manager, 
        model_id, 
        signature_class,
        program_manager, 
        metadata, 
        output_processor or DefaultOutputProcessor(),
        metrics
    )
    
    # Create processor
    processor = ModelProcessor(
        backend=tracked_backend,
        accepted_types=accepted_types,
        output_type=output_type,
        metadata=metadata or {}
    )
    
    # Track processor and return
    return TrackingFactory.track_step(
        processor, 
        metrics, 
        f"{signature_class.__name__}Processor"
    )