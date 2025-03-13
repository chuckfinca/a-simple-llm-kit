import uuid
import datetime
import importlib
from typing import Dict, Any, List, Optional, Type, Union, Tuple
import dspy

from app.core import logging
from app.core.program_registry import ProgramRegistry
from app.core.model_interfaces import Signature
from app.core.types import ProgramMetadata, ProgramExecutionInfo


class ProgramManager:
    """
    Manager for DSPy programs, handling registration, execution tracking, and versioning.
    """
    
    def __init__(
        self,
        model_manager,
        programs_dir: str = "dspy_programs"
    ):
        self.model_manager = model_manager
        self.registry = ProgramRegistry(programs_dir)
        
        # Track execution history
        self.executions: List[ProgramExecutionInfo] = []
        
        # Cache model information from config
        self.model_info = self._extract_model_info()
    
    def _extract_model_info(self) -> Dict[str, Dict[str, str]]:
        """Extract model information from the model config."""
        model_info = {}
        try:
            for model_id, config in self.model_manager.config.get('models', {}).items():
                model_name = config.get('model_name', '')
                provider = model_name.split('/')[0] if '/' in model_name else 'unknown'
                base_model = model_name.split('/')[-1] if '/' in model_name else model_name
                
                model_info[model_id] = {
                    'provider': provider,
                    'base_model': base_model,
                    'model_name': model_name
                }
        except Exception as e:
            logging.error(f"Error extracting model info from config: {e}")
        
        return model_info
    
    def register_program(
        self,
        program_class: Type[Signature],
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0"
    ) -> ProgramMetadata:
        """
        Register a new DSPy program.
        
        Args:
            program_class: The DSPy Signature class
            name: Optional name for the program
            description: Optional description
            tags: Optional list of tags
            version: Version string (semantic versioning recommended)
            
        Returns:
            ProgramMetadata with the registration details
        """
        return self.registry.register_program(
            program_class=program_class,
            name=name or program_class.__name__,
            description=description,
            tags=tags,
            version=version
        )
    
    def get_program(
        self,
        program_id: str,
        version: str = "latest"
    ) -> Optional[Type[Signature]]:
        """
        Get a program by ID and version.
        
        Args:
            program_id: The program ID
            version: Program version or "latest"
            
        Returns:
            The program class or None if not found
        """
        return self.registry.get_program(program_id, version)
    
    async def execute_program(
        self,
        program_id: str,
        model_id: str,
        input_data: Dict[str, Any],
        program_version: str = "latest",
        trace_id: Optional[str] = None
    ) -> Tuple[Any, ProgramExecutionInfo]:
        """
        Execute a program with tracking information.
        
        Args:
            program_id: Program ID to execute
            model_id: Model ID to use
            input_data: Input data dictionary for the program
            program_version: Program version to use (defaults to latest)
            trace_id: Optional trace ID for distributed tracing
            
        Returns:
            Tuple of (program result, execution info)
        """
        # Get the program
        program_class = self.registry.get_program(program_id, program_version)
        if not program_class:
            raise ValueError(f"Program {program_id} version {program_version} not found")
            
        # Get actual version if "latest" was specified
        if program_version == "latest":
            program_metadata = self.registry.get_program_metadata(program_id)
            if program_metadata:
                program_version = program_metadata.version
        
        # Get the model
        lm = self.model_manager.get_model(model_id)
        if not lm:
            raise ValueError(f"Model {model_id} not found")
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Get program metadata for name
        program_metadata = self.registry.get_program_metadata(program_id, program_version)
        program_name = program_metadata.name if program_metadata else program_id
        
        # Get model info
        model_info = self.model_info.get(model_id, {})
        
        # Create execution info
        execution_info = ProgramExecutionInfo(
            program_id=program_id,
            program_version=program_version,
            program_name=program_name,
            model_id=model_id,
            model_info=model_info,
            execution_id=execution_id,
            timestamp=timestamp,
            trace_id=trace_id
        )
        
        # Execute the program with the appropriate model
        try:
            dspy.configure(lm=lm)
            predictor = dspy.Predict(program_class)
            
            # Execute the program
            result = predictor(**input_data)
            
            # Track successful execution
            self.executions.append(execution_info)
            
            return result, execution_info
        except Exception as e:
            logging.error(
                f"Error executing program {program_id}/{program_version} with model {model_id}: {e}",
                exc_info=True
            )
            raise
    
    def get_execution_history(
        self,
        program_id: Optional[str] = None,
        model_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ProgramExecutionInfo]:
        """
        Get execution history, optionally filtered.
        
        Args:
            program_id: Optional program ID filter
            model_id: Optional model ID filter
            limit: Maximum number of history items to return
            
        Returns:
            List of execution info objects
        """
        # Apply filters
        filtered = self.executions
        
        if program_id:
            filtered = [e for e in filtered if e.program_id == program_id]
            
        if model_id:
            filtered = [e for e in filtered if e.model_id == model_id]
        
        # Sort by timestamp (newest first) and limit
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def register_optimized_program(
        self,
        program_class: Type[Signature],
        parent_id: str,
        optimizer_name: str,
        parent_version: str = "latest",
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> ProgramMetadata:
        """
        Register an optimized version of a program.
        
        Args:
            program_class: The optimized program class
            parent_id: Parent program ID
            optimizer_name: Name of the optimizer used
            parent_version: Parent program version
            name: Optional name for the optimized program
            description: Optional description
            tags: Optional tags
            
        Returns:
            Metadata for the registered program
        """
        # Resolve "latest" for parent version
        if parent_version == "latest":
            parent_metadata = self.registry.get_program_metadata(parent_id)
            if not parent_metadata:
                raise ValueError(f"Parent program {parent_id} not found")
            parent_version = parent_metadata.version
        
        return self.registry.register_optimized_program(
            program_class=program_class,
            parent_id=parent_id,
            parent_version=parent_version,
            optimizer_name=optimizer_name,
            name=name,
            description=description,
            tags=tags
        )
    
    def get_program_tree(self, program_id: str) -> Dict[str, Any]:
        """
        Get the tree structure of a program and its derivatives.
        
        Args:
            program_id: Root program ID
            
        Returns:
            Dictionary representation of the program tree
        """
        return self.registry.get_program_tree(program_id)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from config.
        
        Returns:
            List of model information dictionaries
        """
        return [
            {
                "model_id": model_id,
                **info
            }
            for model_id, info in self.model_info.items()
        ]
    
    def save_evaluation_result(
        self,
        program_id: str,
        model_id: str,
        results: Dict[str, Any],
        program_version: str = "latest",
        evaluation_id: Optional[str] = None
    ):
        """
        Save evaluation results for a program-model pair.
        
        Args:
            program_id: Program ID
            model_id: Model ID
            results: Evaluation result dictionary
            program_version: Program version (defaults to latest)
            evaluation_id: Optional evaluation ID (generated if not provided)
        """
        # Resolve "latest" for program version
        if program_version == "latest":
            program_metadata = self.registry.get_program_metadata(program_id)
            if not program_metadata:
                raise ValueError(f"Program {program_id} not found")
            program_version = program_metadata.version
        
        # Get model info
        model_info = self.model_info.get(model_id, {})
        
        # Generate evaluation ID if not provided
        evaluation_id = evaluation_id or str(uuid.uuid4())
        
        self.registry.save_evaluation_result(
            program_id=program_id,
            version=program_version,
            model_id=model_id,
            model_info=model_info,
            evaluation_id=evaluation_id,
            results=results
        )
    
    def get_evaluation_results(
        self,
        program_id: str,
        version: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation results for a program.
        
        Args:
            program_id: Program ID
            version: Optional program version filter
            model_id: Optional model ID filter
            
        Returns:
            List of evaluation result dictionaries
        """
        return self.registry.get_evaluation_results(
            program_id=program_id,
            version=version,
            model_id=model_id
        )