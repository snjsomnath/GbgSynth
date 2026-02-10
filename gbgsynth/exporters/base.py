"""
Base exporter interface for GbgSynth population exports.

All exporters should inherit from BaseExporter and implement
the export() and get_schema() methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from gbgsynth.area import GbgArea


class BaseExporter(ABC):
    """
    Abstract base class for population exporters.
    
    Subclasses must implement:
    - export(): Convert and write population to target format
    - get_schema(): Return documentation of output format
    
    Example:
        >>> class MyExporter(BaseExporter):
        ...     name = "myformat"
        ...     file_extension = ".json"
        ...     
        ...     def export(self, area, output_path):
        ...         # Convert and write data
        ...         pass
        ...     
        ...     def get_schema(self):
        ...         return {"description": "My format"}
    """
    
    # Subclasses should override these
    name: str = "base"
    file_extension: str = ".json"
    description: str = "Base exporter (not for direct use)"
    
    @abstractmethod
    def export(self, area: "GbgArea", output_path: Path) -> Path:
        """
        Export area population to target format.
        
        Args:
            area: GbgArea instance with generated population
            output_path: Path for output file or directory
            
        Returns:
            Path to created output file/directory
            
        Raises:
            DataNotGeneratedError: If area.generate() hasn't been called
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema or documentation of output format.
        
        Returns:
            Dictionary describing the output format, fields, and options
        """
        pass
    
    def validate_area(self, area: "GbgArea") -> None:
        """
        Validate that area has generated data.
        
        Args:
            area: GbgArea to validate
            
        Raises:
            RuntimeError: If generate() hasn't been called
        """
        if not area._is_generated:
            raise RuntimeError(
                f"Cannot export: area.generate() must be called first. "
                f"Use area.generate() before area.export('{self.name}', ...)"
            )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
