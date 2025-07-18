import uuid
import dspy
from typing import Dict, Any, List, Optional, Type, Tuple
from app.core.utils import format_timestamp
from app.core import logging
from app.core.program_registry import ProgramRegistry
from app.core.model_interfaces import Signature
from app.core.types import ProgramMetadata, ProgramExecutionInfo

class ProgramManager:
    """
    Manager for DSPy programs, handling registration, execution tracking, and versioning.
    """
    def __init__(self, model_manager, programs_dir: str = "dspy_programs"):
        self.model_manager = model_manager
        self.registry = ProgramRegistry(programs_dir)
        self.executions: List[ProgramExecutionInfo] = []
        self.model_info = self._extract_model_info()

    def _extract_model_info(self) -> Dict[str, Dict[str, str]]:
        model_info = {}
        try:
            for model_id, config in self.model_manager.config.get('models', {}).items():
                model_name = config.get('model_name', '')
                provider = model_name.split('/')[0] if '/' in model_name else 'unknown'
                base_model = model_name.split('/')[-1] if '/' in model_name else model_name
                model_info[model_id] = {
                    'provider': provider, 'base_model': base_model, 'model_name': model_name
                }
        except Exception as e:
            logging.error(f"Error extracting model info from config: {e}")
        return model_info

    def register_program(
        self, program_class: Type[Signature], name: Optional[str] = None,
        description: Optional[str] = None, tags: Optional[List[str]] = None, version: str = "1.0.0"
    ) -> ProgramMetadata:
        return self.registry.register_program(
            program_class=program_class, name=name or program_class.__name__,
            description=description, tags=tags, version=version
        )

    def get_program(self, program_id: str, version: str = "latest") -> Optional[Type[Signature]]:
        return self.registry.get_program(program_id, version)

    async def execute_program(
            self,
            program_id: str,
            model_id: str,
            input_data: Dict[str, Any],
            program_version: str = "latest",
            trace_id: Optional[str] = None
        ) -> Tuple[Any, ProgramExecutionInfo, Optional[str]]:
            """
            Executes a program and now returns the raw completion text by correctly
            accessing the 'outputs' key in the language model's history.
            """
            program_class = self.registry.get_program(program_id, program_version)
            if not program_class:
                raise ValueError(f"Program {program_id} version {program_version} not found")
    
            if program_version == "latest":
                metadata = self.registry.get_program_metadata(program_id)
                if metadata: program_version = metadata.version
    
            lm = self.model_manager.get_model(model_id)
            if not lm: raise ValueError(f"Model {model_id} not found")
    
            program_metadata = self.registry.get_program_metadata(program_id, program_version)
            execution_info = ProgramExecutionInfo(
                program_id=program_id, program_version=program_version,
                program_name=program_metadata.name if program_metadata else program_id,
                model_id=model_id, model_info=self.model_info.get(model_id, {}),
                execution_id=str(uuid.uuid4()), timestamp=format_timestamp(), trace_id=trace_id
            )
    
            try:
                dspy.configure(lm=lm)
                predictor = dspy.Predict(program_class)
                result = predictor(**input_data)
    
                # --- KEY CHANGE: Extract raw text from the correct 'outputs' key ---
                raw_completion_text = None
                logging.info("Attempting to extract raw completion from LM history...")
                try:
                    if hasattr(lm, 'history') and lm.history:
                        last_interaction = lm.history[-1]
                        # This is the corrected, more direct access path based on your logs.
                        if 'outputs' in last_interaction and last_interaction['outputs']:
                            raw_completion_text = last_interaction['outputs'][0]
                            logging.info("SUCCESS: Extracted raw completion from history['outputs'].")
                        else:
                            logging.warning("History entry found, but 'outputs' key is missing or empty.")
                    else:
                        logging.warning("LM history is missing or empty. Cannot extract raw completion.")
                except Exception as e:
                    logging.error(f"ProgramManager failed to extract raw completion text due to an error: {e}", exc_info=True)
                # --- END KEY CHANGE ---
    
                self.executions.append(execution_info)
                return result, execution_info, raw_completion_text
    
            except Exception as e:
                logging.error(f"Error executing program {program_id}/{program_version}: {e}", exc_info=True)
                raise

    # (The rest of the methods in this file remain the same)
    def get_execution_history(self, program_id: Optional[str] = None, model_id: Optional[str] = None, limit: int = 100) -> List[ProgramExecutionInfo]:
        filtered = self.executions
        if program_id:
            filtered = [e for e in filtered if e.program_id == program_id]
        if model_id:
            filtered = [e for e in filtered if e.model_id == model_id]
        return sorted(filtered, key=lambda e: e.timestamp, reverse=True)[:limit]

    def register_optimized_program(self, program_class: Type[Signature], parent_id: str, optimizer_name: str, parent_version: str = "latest", name: Optional[str] = None, description: Optional[str] = None, tags: Optional[List[str]] = None) -> ProgramMetadata:
        if parent_version == "latest":
            parent_metadata = self.registry.get_program_metadata(parent_id)
            if not parent_metadata: raise ValueError(f"Parent program {parent_id} not found")
            parent_version = parent_metadata.version
        return self.registry.register_optimized_program(
            program_class=program_class, parent_id=parent_id, parent_version=parent_version,
            optimizer_name=optimizer_name, name=name, description=description, tags=tags
        )

    def get_program_tree(self, program_id: str) -> Dict[str, Any]:
        return self.registry.get_program_tree(program_id)

    def get_available_models(self) -> List[Dict[str, Any]]:
        return [{"model_id": model_id, **info} for model_id, info in self.model_info.items()]

    def save_evaluation_result(self, program_id: str, model_id: str, results: Dict[str, Any], program_version: str = "latest", evaluation_id: Optional[str] = None):
        if program_version == "latest":
            program_metadata = self.registry.get_program_metadata(program_id)
            if not program_metadata: raise ValueError(f"Program {program_id} not found")
            program_version = program_metadata.version
        self.registry.save_evaluation_result(
            program_id=program_id, version=program_version, model_id=model_id,
            model_info=self.model_info.get(model_id, {}),
            evaluation_id=evaluation_id or str(uuid.uuid4()), results=results
        )

    def get_evaluation_results(self, program_id: str, version: Optional[str] = None, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.registry.get_evaluation_results(program_id=program_id, version=version, model_id=model_id)