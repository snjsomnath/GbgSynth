"""
Custom exceptions for GbgSynth.

Provides clear, specific error messages for common failure modes.
"""


class GbgSynthError(Exception):
    """Base exception for all GbgSynth errors."""
    pass


class AreaNotFoundError(GbgSynthError):
    """Raised when an area code or name cannot be resolved."""
    
    def __init__(self, area_identifier: str, suggestion: str = None):
        self.area_identifier = area_identifier
        self.suggestion = suggestion
        msg = f"Area '{area_identifier}' not found."
        if suggestion:
            msg += f" {suggestion}"
        else:
            msg += " Use GbgSynth.list_areas() to see available areas."
        super().__init__(msg)


class APIError(GbgSynthError):
    """Raised when the PxWeb API fails or returns an error."""
    
    def __init__(self, message: str, url: str = None, status_code: int = None):
        self.url = url
        self.status_code = status_code
        if status_code:
            message = f"API error (HTTP {status_code}): {message}"
        super().__init__(message)


class DataNotGeneratedError(GbgSynthError):
    """Raised when accessing population data before generate() is called."""
    
    def __init__(self, operation: str = "this operation"):
        super().__init__(
            f"Must call generate() before {operation}. "
            f"Example: area.generate() then area.save()"
        )


class InvalidDataError(GbgSynthError):
    """Raised when input data is malformed or invalid."""
    
    def __init__(self, message: str, field: str = None):
        self.field = field
        if field:
            message = f"Invalid data in '{field}': {message}"
        super().__init__(message)


class PrivacySuppressionError(GbgSynthError):
    """Raised when census data is suppressed for privacy (small populations)."""
    
    def __init__(self, area_name: str, table: str = None):
        self.area_name = area_name
        self.table = table
        msg = f"Data for '{area_name}' is suppressed for privacy (population too small)."
        if table:
            msg += f" Affected table: {table}"
        super().__init__(msg)
