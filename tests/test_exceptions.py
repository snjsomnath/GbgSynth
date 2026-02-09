"""
Tests for custom exceptions.
"""

import pytest
from gbgsynth.exceptions import (
    GbgSynthError,
    AreaNotFoundError,
    APIError,
    DataNotGeneratedError,
    InvalidDataError,
    PrivacySuppressionError,
)


class TestExceptionHierarchy:
    """Test that all exceptions inherit from GbgSynthError."""
    
    def test_area_not_found_inherits(self):
        """AreaNotFoundError should inherit from GbgSynthError."""
        assert issubclass(AreaNotFoundError, GbgSynthError)
    
    def test_api_error_inherits(self):
        """APIError should inherit from GbgSynthError."""
        assert issubclass(APIError, GbgSynthError)
    
    def test_data_not_generated_inherits(self):
        """DataNotGeneratedError should inherit from GbgSynthError."""
        assert issubclass(DataNotGeneratedError, GbgSynthError)
    
    def test_invalid_data_inherits(self):
        """InvalidDataError should inherit from GbgSynthError."""
        assert issubclass(InvalidDataError, GbgSynthError)
    
    def test_privacy_suppression_inherits(self):
        """PrivacySuppressionError should inherit from GbgSynthError."""
        assert issubclass(PrivacySuppressionError, GbgSynthError)


class TestAreaNotFoundError:
    """Test AreaNotFoundError behavior."""
    
    def test_basic_message(self):
        """Test basic error message."""
        err = AreaNotFoundError("Fake Area")
        assert "Fake Area" in str(err)
        assert "not found" in str(err)
    
    def test_includes_hint(self):
        """Test that message includes a hint."""
        err = AreaNotFoundError("Fake Area")
        assert "list_areas()" in str(err)
    
    def test_stores_area_identifier(self):
        """Test that area identifier is stored."""
        err = AreaNotFoundError("Fake Area")
        assert err.area_identifier == "Fake Area"


class TestDataNotGeneratedError:
    """Test DataNotGeneratedError behavior."""
    
    def test_basic_message(self):
        """Test basic error message."""
        err = DataNotGeneratedError("saving")
        assert "generate()" in str(err)
        assert "saving" in str(err)
    
    def test_default_operation(self):
        """Test default operation name."""
        err = DataNotGeneratedError()
        assert "this operation" in str(err)


class TestAPIError:
    """Test APIError behavior."""
    
    def test_basic_message(self):
        """Test basic error message."""
        err = APIError("Connection failed")
        assert "Connection failed" in str(err)
    
    def test_with_status_code(self):
        """Test message with status code."""
        err = APIError("Not found", status_code=404)
        assert "404" in str(err)
    
    def test_stores_url(self):
        """Test that URL is stored."""
        err = APIError("Failed", url="https://example.com/api")
        assert err.url == "https://example.com/api"


class TestInvalidDataError:
    """Test InvalidDataError behavior."""
    
    def test_basic_message(self):
        """Test basic error message."""
        err = InvalidDataError("Data is empty")
        assert "Data is empty" in str(err)
    
    def test_with_field(self):
        """Test message with field name."""
        err = InvalidDataError("cannot be negative", field="population")
        assert "population" in str(err)
    
    def test_stores_field(self):
        """Test that field is stored."""
        err = InvalidDataError("error", field="test_field")
        assert err.field == "test_field"


class TestPrivacySuppressionError:
    """Test PrivacySuppressionError behavior."""
    
    def test_basic_message(self):
        """Test basic error message."""
        err = PrivacySuppressionError("Small Area")
        assert "Small Area" in str(err)
        assert "privacy" in str(err).lower()
    
    def test_with_table(self):
        """Test message with table name."""
        err = PrivacySuppressionError("Small Area", table="income_table")
        assert "income_table" in str(err)
    
    def test_stores_area_name(self):
        """Test that area name is stored."""
        err = PrivacySuppressionError("Test Area")
        assert err.area_name == "Test Area"
