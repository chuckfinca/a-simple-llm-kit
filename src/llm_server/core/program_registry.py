import os
import json
import hashlib
import inspect
import importlib
import datetime
from typing import Dict, Any, List, Optional, Type, Union
from pathlib import Path
from llm_server.core.utils import format_timestamp
import dspy

from llm_server.core import logging
from llm_server.core.model_interfaces import Signature
from llm_server.core.types import ProgramMetadata


class ProgramRegistry:
    """Registry for managing DSPy program signatures and their versions."""
    
    def __init__(self, storage_dir: str = "dspy_programs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Structure to hold loaded programs
        self.programs: Dict[str, Dict[str, Type[Signature]]] = {}
        
        # Load existing programs
        self._load_programs()
        
    def _load_programs(self):
        """Load existing programs from storage directory."""
        if not self.storage_dir.exists():
            return
            
        for program_dir in self.storage_dir.iterdir():
            if program_dir.is_dir():
                program_id = program_dir.name
                self.programs[program_id] = {}
                
                for version_file in program_dir.glob("*.json"):
                    try:
                        with open(version_file, "r") as f:
                            program_data = json.load(f)
                            
                        # Import the module dynamically if it exists
                        module_path = program_data.get("module_path")
                        if module_path:
                            try:
                                module = importlib.import_module(module_path)
                                program_class = getattr(module, program_data["class_name"])
                                self.programs[program_id][version_file.stem] = program_class
                            except (ImportError, AttributeError) as e:
                                logging.warning(
                                    f"Failed to load program {program_id}/{version_file.stem}: {e}"
                                )
                    except Exception as e:
                        logging.error(f"Error loading program metadata: {e}")
    
    def register_program(
        self, 
        program_class: Type[Signature], 
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        parent_id: Optional[str] = None,
        parent_version: Optional[str] = None
    ) -> ProgramMetadata:
        """
        Register a DSPy program signature class.
        
        Args:
            program_class: The DSPy signature class to register
            name: Optional name for the program (defaults to class name)
            description: Optional description
            tags: Optional list of tags
            version: Version identifier (semantic versioning recommended)
            parent_id: Optional parent program ID if this is a derivative
            parent_version: Optional parent version if this is a derivative
            
        Returns:
            ProgramMetadata with the registered program details
        """
        name = name or program_class.__name__
        
        # Create a unique ID based on the name
        program_id = self._generate_program_id(name)
        
        # Get the source code
        source_code = inspect.getsource(program_class)
        
        # Create hash for the program version
        code_hash = hashlib.sha256(source_code.encode()).hexdigest()[:8]
        
        # Prepare metadata
        metadata = {
            "id": program_id,
            "name": name,
            "description": description or "",
            "tags": tags or [],
            "version": version,
            "code_hash": code_hash,
            "parent_id": parent_id,
            "parent_version": parent_version,
            "class_name": program_class.__name__,
            "module_path": program_class.__module__,
            "created_at": format_timestamp(),
            "source_code": source_code
        }
        
        # Ensure program directory exists
        program_dir = self.storage_dir / program_id
        program_dir.mkdir(exist_ok=True)
        
        # Save metadata to file
        version_file = program_dir / f"{version}.json"
        with open(version_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Add to in-memory registry
        if program_id not in self.programs:
            self.programs[program_id] = {}
            
        self.programs[program_id][version] = program_class
        
        return ProgramMetadata(
            id=program_id,
            name=name,
            version=version,
            code_hash=code_hash,
            parent_id=parent_id,
            parent_version=parent_version,
            description=description or "",
            tags=tags or []
        )
    
    def register_optimized_program(
        self,
        program_class: Type[Signature],
        parent_id: str,
        parent_version: str,
        optimizer_name: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ProgramMetadata:
        """
        Register an optimized version of a program.
        
        Args:
            program_class: The optimized DSPy signature class
            parent_id: The ID of the parent program
            parent_version: The version of the parent program
            optimizer_name: Name of the optimizer used
            name: Optional name for the program
            description: Optional description
            tags: Optional list of tags
            
        Returns:
            ProgramMetadata with the registered program details
        """
        # Generate a semantic version based on the parent
        next_version = self._get_next_version(parent_id, parent_version)
        
        # Inherit name if not specified
        if name is None and parent_id in self.programs:
            parent_metadata = self.get_program_metadata(parent_id, parent_version)
            if parent_metadata:
                name = f"{parent_metadata.name}_optimized"
        
        # Add optimizer to tags
        if tags is None:
            tags = []
        tags.append(f"optimizer:{optimizer_name}")
        
        return self.register_program(
            program_class=program_class,
            name=name,
            description=description,
            tags=tags,
            version=next_version,
            parent_id=parent_id,
            parent_version=parent_version
        )
    
    def get_program(self, program_id: str, version: str = "latest") -> Optional[Type[Signature]]:
        """
        Get a program by ID and version.
        
        Args:
            program_id: The program ID
            version: The version to retrieve, or "latest"
            
        Returns:
            The program class or None if not found
        """
        if program_id not in self.programs:
            return None
            
        if version == "latest":
            # Find the latest version by semantic versioning
            versions = sorted(self.programs[program_id].keys(), 
                             key=lambda v: [int(x) for x in v.split(".")])
            if not versions:
                return None
            version = versions[-1]
            
        return self.programs[program_id].get(version)
    
    def get_program_metadata(self, program_id: str, version: str = "latest") -> Optional[ProgramMetadata]:
        """
        Get program metadata by ID and version.
        
        Args:
            program_id: The program ID
            version: The version to retrieve, or "latest"
            
        Returns:
            ProgramMetadata or None if not found
        """
        if program_id not in self.programs:
            return None
            
        if version == "latest":
            # Find the latest version by semantic versioning
            versions = sorted(self.programs[program_id].keys(), 
                             key=lambda v: [int(x) for x in v.split(".")])
            if not versions:
                return None
            version = versions[-1]
        
        version_file = self.storage_dir / program_id / f"{version}.json"
        if not version_file.exists():
            return None
            
        try:
            with open(version_file, "r") as f:
                data = json.load(f)
                
            return ProgramMetadata(
                id=data["id"],
                name=data["name"],
                version=data["version"],
                code_hash=data["code_hash"],
                parent_id=data.get("parent_id"),
                parent_version=data.get("parent_version"),
                description=data.get("description", ""),
                tags=data.get("tags", [])
            )
        except Exception as e:
            logging.error(f"Error loading program metadata: {e}")
            return None
    
    def list_programs(self, tags: Optional[List[str]] = None) -> List[ProgramMetadata]:
        """
        List all registered programs, optionally filtered by tags.
        
        Args:
            tags: Optional list of tags to filter by
            
        Returns:
            List of ProgramMetadata
        """
        result = []
        
        for program_id in self.programs:
            # Get the latest version for each program
            metadata = self.get_program_metadata(program_id)
            if metadata:
                # Apply tag filtering if specified
                if tags and not all(tag in metadata.tags for tag in tags):
                    continue
                    
                result.append(metadata)
                
        return result
    
    def list_program_versions(self, program_id: str) -> List[ProgramMetadata]:
        """
        List all versions of a specific program.
        
        Args:
            program_id: The program ID
            
        Returns:
            List of ProgramMetadata for each version
        """
        if program_id not in self.programs:
            return []
            
        result = []
        for version in self.programs[program_id].keys():
            metadata = self.get_program_metadata(program_id, version)
            if metadata:
                result.append(metadata)
                
        return sorted(result, key=lambda x: [int(v) for v in x.version.split(".")])
        
    def get_program_tree(self, program_id: str) -> Dict[str, Any]:
        """
        Get the full tree of a program and its derivatives.
        
        Args:
            program_id: The root program ID
            
        Returns:
            Dictionary representing the program tree
        """
        root_metadata = self.get_program_metadata(program_id)
        if not root_metadata:
            return {}
            
        tree = {
            "metadata": root_metadata.model_dump(),
            "versions": {},
            "derivatives": []
        }
        
        # Add all versions
        for version_metadata in self.list_program_versions(program_id):
            tree["versions"][version_metadata.version] = version_metadata.model_dump()
        
        # Add all derivatives (any program that has this as a parent)
        for candidate_id in self.programs:
            # Skip self
            if candidate_id == program_id:
                continue
                
            for version in self.programs[candidate_id]:
                metadata = self.get_program_metadata(candidate_id, version)
                if metadata and metadata.parent_id == program_id:
                    derivative_tree = self.get_program_tree(candidate_id)
                    if derivative_tree:
                        tree["derivatives"].append(derivative_tree)
                    break
                    
        return tree
    
    def save_evaluation_result(
        self, 
        program_id: str, 
        version: str,
        model_id: str,
        model_info: Dict[str, Any],
        evaluation_id: str,
        results: Dict[str, Any]
    ):
        """
        Save evaluation results for a program-model pair.
        
        Args:
            program_id: The program ID
            version: The program version
            model_id: The model ID
            model_info: Model information dictionary
            evaluation_id: Unique identifier for the evaluation
            results: Evaluation results
        """
        eval_dir = self.storage_dir / program_id / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        # Create a unique filename for this evaluation
        filename = f"{evaluation_id}_{model_id}.json"
        
        eval_data = {
            "program_id": program_id,
            "program_version": version,
            "model_id": model_id,
            "model_info": model_info,
            "evaluation_id": evaluation_id,
            "timestamp": format_timestamp(),
            "results": results
        }
        
        with open(eval_dir / filename, "w") as f:
            json.dump(eval_data, f, indent=2)
    
    def get_evaluation_results(
        self, 
        program_id: str, 
        version: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation results, optionally filtered.
        
        Args:
            program_id: The program ID
            version: Optional program version to filter by
            model_id: Optional model ID to filter by
            
        Returns:
            List of evaluation result dictionaries
        """
        eval_dir = self.storage_dir / program_id / "evaluations"
        if not eval_dir.exists():
            return []
            
        results = []
        for eval_file in eval_dir.glob("*.json"):
            try:
                with open(eval_file, "r") as f:
                    data = json.load(f)
                    
                # Apply filters
                if version and data["program_version"] != version:
                    continue
                if model_id and data["model_id"] != model_id:
                    continue
                    
                results.append(data)
            except Exception as e:
                logging.error(f"Error loading evaluation result: {e}")
                
        return results
    
    def _generate_program_id(self, name: str) -> str:
        """Generate a unique program ID based on the name."""
        # Replace spaces with underscores and remove special characters
        base_id = "".join(c if c.isalnum() or c == "_" else "" for c in name.replace(" ", "_"))
        base_id = base_id.lower()
        
        # Check if the ID already exists, append a number if needed
        if base_id not in self.programs:
            return base_id
            
        # Find the next available number
        i = 1
        while f"{base_id}_{i}" in self.programs:
            i += 1
            
        return f"{base_id}_{i}"
    
    def _get_next_version(self, program_id: str, parent_version: str) -> str:
        """Generate the next semantic version for a derived program."""
        if program_id not in self.programs:
            return "1.0.0"
            
        try:
            # Parse the parent version
            major, minor, patch = map(int, parent_version.split("."))
            
            # Increment the minor version for an optimization
            return f"{major}.{minor + 1}.0"
        except ValueError:
            # If parent_version isn't a valid semantic version, start at 1.0.0
            return "1.0.0"