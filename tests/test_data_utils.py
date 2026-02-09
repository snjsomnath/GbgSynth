"""
Tests for data_utils module - shapefile download and management.
"""

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from gbgsynth.data_utils import (
    SHAPEFILE_EXTENSIONS,
    PRI_SHAPEFILE_URL,
    SKIP_NEIGHBOURHOODS,
    is_shapefile_available,
    download_pri_shapefile,
    ensure_shapefile_available,
    get_shapefile_path,
    cleanup_shapefile,
    _extract_and_rename_shapefile,
    is_areas_json_available,
    generate_areas_json,
    ensure_areas_json_available,
    get_areas_json_path,
    ensure_data_available,
    is_footprints_available,
    get_shapefile_bounds,
    download_footprints,
    ensure_footprints_available,
    get_footprints_path,
    is_neighbourhood_heights_available,
    get_missing_neighbourhood_heights,
    generate_neighbourhood_heights,
    ensure_neighbourhood_heights_available,
    get_neighbourhood_heights_path,
    get_neighbourhood_heights_dir,
)


class TestIsShapefileAvailable:
    """Tests for is_shapefile_available function."""

    def test_returns_false_when_directory_missing(self, tmp_path):
        """Should return False when pri_shp directory doesn't exist."""
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', tmp_path / 'nonexistent'):
            assert is_shapefile_available() is False

    def test_returns_false_when_files_missing(self, tmp_path):
        """Should return False when shapefile components are missing."""
        pri_dir = tmp_path / 'pri_shp'
        pri_dir.mkdir()
        # Only create some files
        (pri_dir / 'pri.shp').touch()
        (pri_dir / 'pri.dbf').touch()
        # Missing .shx and .prj
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            assert is_shapefile_available() is False

    def test_returns_true_when_all_files_present(self, tmp_path):
        """Should return True when all shapefile components exist."""
        pri_dir = tmp_path / 'pri_shp'
        pri_dir.mkdir()
        for ext in SHAPEFILE_EXTENSIONS:
            (pri_dir / f'pri{ext}').touch()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            assert is_shapefile_available() is True


class TestDownloadPriShapefile:
    """Tests for download_pri_shapefile function."""

    def _create_mock_zip(self) -> bytes:
        """Create a mock ZIP file containing shapefile components."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            # Add mock shapefile components with original names
            zf.writestr('Primärområde.shp', b'mock shp content')
            zf.writestr('Primärområde.shx', b'mock shx content')
            zf.writestr('Primärområde.dbf', b'mock dbf content')
            zf.writestr('Primärområde.prj', b'mock prj content')
        return buffer.getvalue()

    def test_downloads_and_extracts_shapefile(self, tmp_path):
        """Should download ZIP and extract shapefile components."""
        target_dir = tmp_path / 'pri_shp'
        mock_response = Mock()
        mock_response.content = self._create_mock_zip()
        mock_response.raise_for_status = Mock()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get', return_value=mock_response):
                result = download_pri_shapefile(target_dir=target_dir)
        
        assert result is True
        assert target_dir.exists()
        assert (target_dir / 'pri.shp').exists()
        assert (target_dir / 'pri.shx').exists()
        assert (target_dir / 'pri.dbf').exists()
        assert (target_dir / 'pri.prj').exists()

    def test_renames_files_correctly(self, tmp_path):
        """Should rename files to standard 'pri.*' names."""
        target_dir = tmp_path / 'pri_shp'
        mock_response = Mock()
        mock_response.content = self._create_mock_zip()
        mock_response.raise_for_status = Mock()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get', return_value=mock_response):
                download_pri_shapefile(target_dir=target_dir)
        
        # Verify content was preserved during rename
        assert (target_dir / 'pri.shp').read_bytes() == b'mock shp content'

    def test_skips_download_when_already_available(self, tmp_path):
        """Should skip download if shapefile already exists."""
        target_dir = tmp_path / 'pri_shp'
        target_dir.mkdir()
        for ext in SHAPEFILE_EXTENSIONS:
            (target_dir / f'pri{ext}').touch()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('requests.get') as mock_get:
                result = download_pri_shapefile(target_dir=target_dir, force=False)
        
        assert result is True
        mock_get.assert_not_called()

    def test_force_redownload(self, tmp_path):
        """Should re-download when force=True even if files exist."""
        target_dir = tmp_path / 'pri_shp'
        target_dir.mkdir()
        for ext in SHAPEFILE_EXTENSIONS:
            (target_dir / f'pri{ext}').write_text('old content')
        
        mock_response = Mock()
        mock_response.content = self._create_mock_zip()
        mock_response.raise_for_status = Mock()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get', return_value=mock_response):
                result = download_pri_shapefile(target_dir=target_dir, force=True)
        
        assert result is True
        # Verify new content was written
        assert (target_dir / 'pri.shp').read_bytes() == b'mock shp content'

    def test_raises_on_http_error(self, tmp_path):
        """Should raise RequestException on HTTP errors."""
        target_dir = tmp_path / 'pri_shp'
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get') as mock_get:
                mock_get.return_value.raise_for_status.side_effect = requests.HTTPError("404")
                
                with pytest.raises(requests.HTTPError):
                    download_pri_shapefile(target_dir=target_dir)

    def test_raises_on_invalid_zip(self, tmp_path):
        """Should raise BadZipFile on invalid ZIP content."""
        target_dir = tmp_path / 'pri_shp'
        mock_response = Mock()
        mock_response.content = b'not a zip file'
        mock_response.raise_for_status = Mock()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get', return_value=mock_response):
                with pytest.raises(zipfile.BadZipFile):
                    download_pri_shapefile(target_dir=target_dir)

    def test_returns_false_on_empty_zip(self, tmp_path):
        """Should return False if ZIP contains no shapefile components."""
        target_dir = tmp_path / 'pri_shp'
        
        # Create empty ZIP
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr('readme.txt', b'no shapefile here')
        
        mock_response = Mock()
        mock_response.content = buffer.getvalue()
        mock_response.raise_for_status = Mock()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get', return_value=mock_response):
                result = download_pri_shapefile(target_dir=target_dir)
        
        assert result is False

    def test_uses_correct_url(self, tmp_path):
        """Should use the correct Gothenburg city URL by default."""
        target_dir = tmp_path / 'pri_shp'
        mock_response = Mock()
        mock_response.content = self._create_mock_zip()
        mock_response.raise_for_status = Mock()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get', return_value=mock_response) as mock_get:
                download_pri_shapefile(target_dir=target_dir)
        
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert called_url == PRI_SHAPEFILE_URL

    def test_ignores_macosx_folder(self, tmp_path):
        """Should ignore __MACOSX folder in ZIP files."""
        target_dir = tmp_path / 'pri_shp'
        
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr('Primärområde.shp', b'good content')
            zf.writestr('Primärområde.shx', b'good content')
            zf.writestr('Primärområde.dbf', b'good content')
            zf.writestr('Primärområde.prj', b'good content')
            zf.writestr('__MACOSX/Primärområde.shp', b'bad content')
        
        mock_response = Mock()
        mock_response.content = buffer.getvalue()
        mock_response.raise_for_status = Mock()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', target_dir):
            with patch('gbgsynth.data_utils.requests.get', return_value=mock_response):
                download_pri_shapefile(target_dir=target_dir)
        
        # Should use the correct file, not __MACOSX version
        assert (target_dir / 'pri.shp').read_bytes() == b'good content'


class TestEnsureShapefileAvailable:
    """Tests for ensure_shapefile_available function."""

    def test_returns_true_when_available(self, tmp_path):
        """Should return True immediately if shapefile exists."""
        pri_dir = tmp_path / 'pri_shp'
        pri_dir.mkdir()
        for ext in SHAPEFILE_EXTENSIONS:
            (pri_dir / f'pri{ext}').touch()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            assert ensure_shapefile_available() is True

    def test_downloads_when_missing_and_auto_download_true(self, tmp_path):
        """Should download shapefile when missing and auto_download is True."""
        pri_dir = tmp_path / 'pri_shp'
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            with patch('gbgsynth.data_utils.download_pri_shapefile', return_value=True) as mock_dl:
                result = ensure_shapefile_available(auto_download=True)
        
        assert result is True
        mock_dl.assert_called_once()

    def test_returns_false_when_missing_and_auto_download_false(self, tmp_path):
        """Should return False when missing and auto_download is False."""
        pri_dir = tmp_path / 'pri_shp'
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            result = ensure_shapefile_available(auto_download=False)
        
        assert result is False

    def test_handles_download_failure(self, tmp_path):
        """Should return False and log error on download failure."""
        pri_dir = tmp_path / 'pri_shp'
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            with patch('gbgsynth.data_utils.download_pri_shapefile') as mock_dl:
                mock_dl.side_effect = requests.RequestException("Network error")
                result = ensure_shapefile_available(auto_download=True)
        
        assert result is False


class TestGetShapefilePath:
    """Tests for get_shapefile_path function."""

    def test_returns_path_when_exists(self, tmp_path):
        """Should return path when shapefile exists."""
        pri_dir = tmp_path / 'pri_shp'
        pri_dir.mkdir()
        shp_file = pri_dir / 'pri.shp'
        shp_file.touch()
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            result = get_shapefile_path()
        
        assert result == shp_file

    def test_returns_none_when_missing(self, tmp_path):
        """Should return None when shapefile doesn't exist."""
        pri_dir = tmp_path / 'nonexistent'
        
        with patch('gbgsynth.data_utils._PRI_SHP_DIR', pri_dir):
            result = get_shapefile_path()
        
        assert result is None


class TestCleanupShapefile:
    """Tests for cleanup_shapefile function."""

    def test_removes_directory(self, tmp_path):
        """Should remove the shapefile directory."""
        pri_dir = tmp_path / 'pri_shp'
        pri_dir.mkdir()
        (pri_dir / 'pri.shp').touch()
        (pri_dir / 'pri.dbf').touch()
        
        result = cleanup_shapefile(target_dir=pri_dir)
        
        assert result is True
        assert not pri_dir.exists()

    def test_returns_true_when_directory_missing(self, tmp_path):
        """Should return True if directory already doesn't exist."""
        pri_dir = tmp_path / 'nonexistent'
        
        result = cleanup_shapefile(target_dir=pri_dir)
        
        assert result is True


class TestExtractAndRenameShapefile:
    """Tests for _extract_and_rename_shapefile helper function."""

    def test_extracts_all_components(self, tmp_path):
        """Should extract all shapefile components."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr('Area.shp', b'shp data')
            zf.writestr('Area.shx', b'shx data')
            zf.writestr('Area.dbf', b'dbf data')
            zf.writestr('Area.prj', b'prj data')
        
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as zf:
            extracted = _extract_and_rename_shapefile(
                zf, zf.namelist(), tmp_path
            )
        
        assert len(extracted) == 4
        assert (tmp_path / 'pri.shp').exists()
        assert (tmp_path / 'pri.shx').exists()
        assert (tmp_path / 'pri.dbf').exists()
        assert (tmp_path / 'pri.prj').exists()

    def test_handles_case_insensitive_extensions(self, tmp_path):
        """Should handle uppercase extensions."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as zf:
            zf.writestr('Area.SHP', b'shp data')
            zf.writestr('Area.SHX', b'shx data')
            zf.writestr('Area.DBF', b'dbf data')
            zf.writestr('Area.PRJ', b'prj data')
        
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as zf:
            extracted = _extract_and_rename_shapefile(
                zf, zf.namelist(), tmp_path
            )
        
        assert len(extracted) == 4


class TestPriShapefileUrl:
    """Tests for the shapefile URL constant."""

    def test_url_is_valid_format(self):
        """Should be a valid HTTPS URL."""
        assert PRI_SHAPEFILE_URL.startswith('https://')
        assert 'goteborg.se' in PRI_SHAPEFILE_URL

    def test_url_contains_expected_path(self):
        """Should point to the primärområde shapefile."""
        assert 'shp' in PRI_SHAPEFILE_URL.lower()


# ============================================================================
# Areas JSON Tests
# ============================================================================

class TestIsAreasJsonAvailable:
    """Tests for is_areas_json_available function."""

    def test_returns_false_when_file_missing(self, tmp_path):
        """Should return False when areas.json doesn't exist."""
        with patch('gbgsynth.data_utils._AREAS_JSON', tmp_path / 'nonexistent.json'):
            assert is_areas_json_available() is False

    def test_returns_true_when_file_exists(self, tmp_path):
        """Should return True when areas.json exists."""
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{}')
        
        with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
            assert is_areas_json_available() is True

    def test_accepts_custom_path(self, tmp_path):
        """Should check custom path when provided."""
        custom_path = tmp_path / 'custom_areas.json'
        custom_path.write_text('{}')
        
        assert is_areas_json_available(areas_json_path=custom_path) is True
        assert is_areas_json_available(areas_json_path=tmp_path / 'missing.json') is False


class TestGenerateAreasJson:
    """Tests for generate_areas_json function."""

    @pytest.fixture
    def mock_geodataframe(self):
        """Create a mock GeoDataFrame with area data."""
        import pandas as pd
        
        data = {
            'PRIMÄROMRÅ': ['101', '102', '107'],
            'PRIMÄRNAMN': ['Kungsladugård', 'Sanna', 'Haga'],
            'geometry': [None, None, None]  # Simplified for testing
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_gpd(self, mock_geodataframe):
        """Create a mock geopandas module."""
        mock_geopandas = MagicMock()
        mock_geopandas.read_file = MagicMock(return_value=mock_geodataframe)
        return mock_geopandas

    def test_generates_areas_json(self, tmp_path, mock_geodataframe, mock_gpd):
        """Should generate areas.json from shapefile."""
        output_path = tmp_path / 'areas.json'
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
            with patch.dict('sys.modules', {'geopandas': mock_gpd}):
                result = generate_areas_json(
                    shapefile_path=shapefile_path,
                    output_path=output_path
                )
        
        assert result is True
        assert output_path.exists()
        
        import json
        with open(output_path) as f:
            areas = json.load(f)
        
        assert '101' in areas
        assert areas['101']['name'] == 'Kungsladugård'
        assert areas['101']['full'] == '101 Kungsladugård'
        assert '107' in areas
        assert areas['107']['name'] == 'Haga'

    def test_skips_when_file_exists(self, tmp_path):
        """Should skip generation when areas.json already exists."""
        output_path = tmp_path / 'areas.json'
        output_path.write_text('{"existing": "data"}')
        
        # No need to mock geopandas since it should return early
        result = generate_areas_json(output_path=output_path, force=False)
        
        assert result is True
        # File should be unchanged
        assert 'existing' in output_path.read_text()

    def test_force_regeneration(self, tmp_path, mock_geodataframe, mock_gpd):
        """Should regenerate when force=True even if file exists."""
        output_path = tmp_path / 'areas.json'
        output_path.write_text('{"old": "data"}')
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
            with patch.dict('sys.modules', {'geopandas': mock_gpd}):
                result = generate_areas_json(
                    shapefile_path=shapefile_path,
                    output_path=output_path,
                    force=True
                )
        
        assert result is True
        import json
        with open(output_path) as f:
            areas = json.load(f)
        assert 'old' not in areas
        assert '101' in areas

    def test_returns_false_when_shapefile_missing(self, tmp_path):
        """Should return False when shapefile is not available."""
        output_path = tmp_path / 'areas.json'
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=None):
            result = generate_areas_json(output_path=output_path)
        
        assert result is False
        assert not output_path.exists()

    def test_raises_when_geopandas_not_installed(self, tmp_path):
        """Should raise ImportError when geopandas is not available."""
        output_path = tmp_path / 'areas.json'
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
            with patch.dict('sys.modules', {'geopandas': None}):
                # Simulate import failure
                import builtins
                real_import = builtins.__import__
                
                def mock_import(name, *args, **kwargs):
                    if name == 'geopandas':
                        raise ImportError("No module named 'geopandas'")
                    return real_import(name, *args, **kwargs)
                
                with patch.object(builtins, '__import__', mock_import):
                    with pytest.raises(ImportError, match="geopandas"):
                        generate_areas_json(
                            shapefile_path=shapefile_path,
                            output_path=output_path
                        )

    def test_sorts_areas_by_code(self, tmp_path):
        """Should sort areas by numeric code."""
        import pandas as pd
        
        # Create unsorted data
        unsorted_data = {
            'PRIMÄROMRÅ': ['302', '101', '205'],
            'PRIMÄRNAMN': ['Utby', 'Kungsladugård', 'Skår'],
            'geometry': [None, None, None]
        }
        unsorted_df = pd.DataFrame(unsorted_data)
        
        # Create mock geopandas
        mock_gpd = MagicMock()
        mock_gpd.read_file = MagicMock(return_value=unsorted_df)
        
        output_path = tmp_path / 'areas.json'
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
            with patch.dict('sys.modules', {'geopandas': mock_gpd}):
                generate_areas_json(
                    shapefile_path=shapefile_path,
                    output_path=output_path
                )
        
        import json
        with open(output_path) as f:
            areas = json.load(f)
        
        # Check order
        keys = list(areas.keys())
        assert keys == ['101', '205', '302']


class TestEnsureAreasJsonAvailable:
    """Tests for ensure_areas_json_available function."""

    def test_returns_true_when_available(self, tmp_path):
        """Should return True immediately if areas.json exists."""
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{}')
        
        with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
            assert ensure_areas_json_available() is True

    def test_generates_when_missing_and_auto_generate_true(self, tmp_path):
        """Should generate areas.json when missing and auto_generate is True."""
        areas_json = tmp_path / 'areas.json'
        
        with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
            with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=True):
                with patch('gbgsynth.data_utils.generate_areas_json', return_value=True) as mock_gen:
                    result = ensure_areas_json_available(auto_generate=True)
        
        assert result is True
        mock_gen.assert_called_once()

    def test_returns_false_when_missing_and_auto_generate_false(self, tmp_path):
        """Should return False when missing and auto_generate is False."""
        areas_json = tmp_path / 'areas.json'
        
        with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
            result = ensure_areas_json_available(auto_generate=False)
        
        assert result is False

    def test_returns_false_when_shapefile_unavailable(self, tmp_path):
        """Should return False when shapefile is not available."""
        areas_json = tmp_path / 'areas.json'
        
        with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
            with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=False):
                result = ensure_areas_json_available(auto_generate=True)
        
        assert result is False


class TestGetAreasJsonPath:
    """Tests for get_areas_json_path function."""

    def test_returns_path_when_exists(self, tmp_path):
        """Should return path when areas.json exists."""
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{}')
        
        with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
            result = get_areas_json_path()
        
        assert result == areas_json

    def test_returns_none_when_missing(self, tmp_path):
        """Should return None when areas.json doesn't exist."""
        areas_json = tmp_path / 'nonexistent.json'
        
        with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
            result = get_areas_json_path()
        
        assert result is None


class TestEnsureDataAvailable:
    """Tests for ensure_data_available function."""

    def test_ensures_both_shapefile_and_areas_json(self, tmp_path):
        """Should ensure both shapefile and areas.json are available."""
        with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=True) as mock_shp:
            with patch('gbgsynth.data_utils.ensure_areas_json_available', return_value=True) as mock_areas:
                with patch('gbgsynth.data_utils.is_footprints_available', return_value=True):
                    result = ensure_data_available()
        
        assert result is True
        mock_shp.assert_called_once_with(auto_download=True)
        mock_areas.assert_called_once_with(auto_generate=True)

    def test_returns_false_if_shapefile_fails(self):
        """Should return False if shapefile is not available."""
        with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=False):
            result = ensure_data_available()
        
        assert result is False

    def test_returns_false_if_areas_json_fails(self):
        """Should return False if areas.json generation fails."""
        with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=True):
            with patch('gbgsynth.data_utils.ensure_areas_json_available', return_value=False):
                result = ensure_data_available()
        
        assert result is False

    def test_respects_auto_download_flag(self):
        """Should pass auto_download flag to ensure_shapefile_available."""
        with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=True) as mock_shp:
            with patch('gbgsynth.data_utils.ensure_areas_json_available', return_value=True):
                with patch('gbgsynth.data_utils.is_footprints_available', return_value=True):
                    ensure_data_available(auto_download=False)
        
        mock_shp.assert_called_once_with(auto_download=False)

    def test_respects_auto_generate_flag(self):
        """Should pass auto_generate flag to ensure_areas_json_available."""
        with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=True):
            with patch('gbgsynth.data_utils.ensure_areas_json_available', return_value=True) as mock_areas:
                with patch('gbgsynth.data_utils.is_footprints_available', return_value=True):
                    ensure_data_available(auto_generate=False)
        
        mock_areas.assert_called_once_with(auto_generate=False)


# ============================================================================
# Footprints Tests
# ============================================================================

class TestIsFootprintsAvailable:
    """Tests for is_footprints_available function."""

    def test_returns_false_when_file_missing(self, tmp_path):
        """Should return False when footprints.gpkg doesn't exist."""
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', tmp_path / 'nonexistent.gpkg'):
            assert is_footprints_available() is False

    def test_returns_true_when_file_exists(self, tmp_path):
        """Should return True when footprints.gpkg exists."""
        footprints_gpkg = tmp_path / 'footprints.gpkg'
        footprints_gpkg.write_bytes(b'mock gpkg content')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            assert is_footprints_available() is True

    def test_accepts_custom_path(self, tmp_path):
        """Should check custom path when provided."""
        custom_path = tmp_path / 'custom_footprints.gpkg'
        custom_path.write_bytes(b'mock gpkg content')
        
        assert is_footprints_available(footprints_path=custom_path) is True
        assert is_footprints_available(footprints_path=tmp_path / 'missing.gpkg') is False


class TestGetShapefileBounds:
    """Tests for get_shapefile_bounds function."""

    def test_returns_bounds_in_epsg_3006(self, tmp_path):
        """Should return bounds tuple reprojected to EPSG:3006."""
        # Create mock CRS that is not 3006
        mock_crs = MagicMock()
        mock_crs.to_epsg.return_value = 4326  # WGS84
        
        # Create reprojected GeoDataFrame
        mock_reprojected_gdf = MagicMock()
        mock_reprojected_gdf.total_bounds = [319000.0, 6397000.0, 330000.0, 6408000.0]
        
        # Create original GeoDataFrame
        mock_gdf = MagicMock()
        mock_gdf.crs = mock_crs
        mock_gdf.to_crs.return_value = mock_reprojected_gdf
        
        mock_gpd = MagicMock()
        mock_gpd.read_file = MagicMock(return_value=mock_gdf)
        
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
            with patch.dict('sys.modules', {'geopandas': mock_gpd}):
                bounds = get_shapefile_bounds()
        
        # Should have called to_crs with epsg=3006
        mock_gdf.to_crs.assert_called_once_with(epsg=3006)
        assert bounds == (319000.0, 6397000.0, 330000.0, 6408000.0)

    def test_skips_reprojection_when_already_epsg_3006(self, tmp_path):
        """Should not reproject if already in EPSG:3006."""
        mock_crs = MagicMock()
        mock_crs.to_epsg.return_value = 3006
        
        mock_gdf = MagicMock()
        mock_gdf.crs = mock_crs
        mock_gdf.total_bounds = [319000.0, 6397000.0, 330000.0, 6408000.0]
        
        mock_gpd = MagicMock()
        mock_gpd.read_file = MagicMock(return_value=mock_gdf)
        
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
            with patch.dict('sys.modules', {'geopandas': mock_gpd}):
                bounds = get_shapefile_bounds()
        
        # Should NOT have called to_crs
        mock_gdf.to_crs.assert_not_called()
        assert bounds == (319000.0, 6397000.0, 330000.0, 6408000.0)

    def test_handles_no_crs(self, tmp_path):
        """Should handle shapefile with no CRS defined."""
        mock_gdf = MagicMock()
        mock_gdf.crs = None
        mock_gdf.total_bounds = [319000.0, 6397000.0, 330000.0, 6408000.0]
        
        mock_gpd = MagicMock()
        mock_gpd.read_file = MagicMock(return_value=mock_gdf)
        
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
            with patch.dict('sys.modules', {'geopandas': mock_gpd}):
                bounds = get_shapefile_bounds()
        
        # Should assume EPSG:3006 and not call to_crs
        mock_gdf.to_crs.assert_not_called()
        assert bounds == (319000.0, 6397000.0, 330000.0, 6408000.0)

    def test_returns_none_when_shapefile_missing(self):
        """Should return None when shapefile is not available."""
        with patch('gbgsynth.data_utils.get_shapefile_path', return_value=None):
            result = get_shapefile_bounds()
        
        assert result is None


class TestDownloadFootprints:
    """Tests for download_footprints function."""

    def test_skips_when_file_exists(self, tmp_path):
        """Should skip download when footprints.gpkg already exists."""
        output_path = tmp_path / 'footprints.gpkg'
        output_path.write_bytes(b'existing content')
        
        result = download_footprints(output_path=output_path, force=False)
        
        assert result is True
        # File should be unchanged
        assert output_path.read_bytes() == b'existing content'

    def test_returns_false_when_bounds_unavailable(self, tmp_path):
        """Should return False when shapefile bounds are not available."""
        output_path = tmp_path / 'footprints.gpkg'
        
        with patch('gbgsynth.data_utils.get_shapefile_bounds', return_value=None):
            result = download_footprints(output_path=output_path)
        
        assert result is False

    def test_raises_when_dtcc_not_installed(self, tmp_path):
        """Should raise ImportError when dtcc is not available."""
        output_path = tmp_path / 'footprints.gpkg'
        bounds = (11.8, 57.6, 12.1, 57.8)
        
        with patch('gbgsynth.data_utils.get_shapefile_bounds', return_value=bounds):
            import builtins
            real_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if name == 'dtcc':
                    raise ImportError("No module named 'dtcc'")
                return real_import(name, *args, **kwargs)
            
            with patch.object(builtins, '__import__', mock_import):
                with pytest.raises(ImportError, match="dtcc"):
                    download_footprints(output_path=output_path, bounds=bounds)

    def test_uses_provided_bounds(self, tmp_path):
        """Should use provided bounds instead of shapefile bounds."""
        output_path = tmp_path / 'footprints.gpkg'
        custom_bounds = (11.9, 57.65, 12.0, 57.75)
        
        # Create mock dtcc module
        mock_dtcc = MagicMock()
        mock_dtcc_bounds = MagicMock()
        mock_dtcc.Bounds = MagicMock(return_value=mock_dtcc_bounds)
        
        # Mock the download to return a GeoDataFrame-like object
        mock_footprints = MagicMock()
        mock_footprints.to_file = MagicMock()
        mock_dtcc.download_footprints = MagicMock(return_value=mock_footprints)
        
        with patch.dict('sys.modules', {'dtcc': mock_dtcc}):
            with patch('gbgsynth.data_utils.get_shapefile_bounds') as mock_get_bounds:
                download_footprints(output_path=output_path, bounds=custom_bounds)
        
        # Should not call get_shapefile_bounds when bounds are provided
        mock_get_bounds.assert_not_called()
        
        # Should create Bounds with provided values
        mock_dtcc.Bounds.assert_called_once_with(
            xmin=custom_bounds[0],
            ymin=custom_bounds[1],
            xmax=custom_bounds[2],
            ymax=custom_bounds[3]
        )


class TestEnsureFootprintsAvailable:
    """Tests for ensure_footprints_available function."""

    def test_returns_true_when_available(self, tmp_path):
        """Should return True immediately if footprints.gpkg exists."""
        footprints_gpkg = tmp_path / 'footprints.gpkg'
        footprints_gpkg.write_bytes(b'mock content')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            assert ensure_footprints_available() is True

    def test_downloads_when_missing_and_auto_download_true(self, tmp_path):
        """Should download footprints when missing and auto_download is True."""
        footprints_gpkg = tmp_path / 'footprints.gpkg'
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=True):
                with patch('gbgsynth.data_utils.download_footprints', return_value=True) as mock_dl:
                    result = ensure_footprints_available(auto_download=True)
        
        assert result is True
        mock_dl.assert_called_once()

    def test_returns_false_when_missing_and_auto_download_false(self, tmp_path):
        """Should return False when missing and auto_download is False."""
        footprints_gpkg = tmp_path / 'footprints.gpkg'
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            result = ensure_footprints_available(auto_download=False)
        
        assert result is False

    def test_returns_false_when_shapefile_unavailable(self, tmp_path):
        """Should return False when shapefile is not available."""
        footprints_gpkg = tmp_path / 'footprints.gpkg'
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=False):
                result = ensure_footprints_available(auto_download=True)
        
        assert result is False

    def test_handles_import_error(self, tmp_path):
        """Should return False when dtcc is not installed."""
        footprints_gpkg = tmp_path / 'footprints.gpkg'
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            with patch('gbgsynth.data_utils.ensure_shapefile_available', return_value=True):
                with patch('gbgsynth.data_utils.download_footprints') as mock_dl:
                    mock_dl.side_effect = ImportError("dtcc not available")
                    result = ensure_footprints_available(auto_download=True)
        
        assert result is False


class TestGetFootprintsPath:
    """Tests for get_footprints_path function."""

    def test_returns_path_when_exists(self, tmp_path):
        """Should return path when footprints.gpkg exists."""
        footprints_gpkg = tmp_path / 'footprints.gpkg'
        footprints_gpkg.write_bytes(b'mock content')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            result = get_footprints_path()
        
        assert result == footprints_gpkg

    def test_returns_none_when_missing(self, tmp_path):
        """Should return None when footprints.gpkg doesn't exist."""
        footprints_gpkg = tmp_path / 'nonexistent.gpkg'
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_GPKG', footprints_gpkg):
            result = get_footprints_path()
        
        assert result is None


# ============================================================================
# Neighbourhood Heights Tests
# ============================================================================

class TestIsNeighbourhoodHeightsAvailable:
    """Tests for is_neighbourhood_heights_available function."""

    def test_returns_false_when_directory_missing(self, tmp_path):
        """Should return False when footprints directory doesn't exist."""
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', tmp_path / 'nonexistent'):
            assert is_neighbourhood_heights_available() is False

    def test_returns_true_for_single_neighbourhood(self, tmp_path):
        """Should return True when specific neighbourhood height file exists."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        (footprints_dir / 'Haga_heights.gpkg').write_bytes(b'mock content')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            assert is_neighbourhood_heights_available(neighbourhood_name='Haga') is True

    def test_returns_false_for_missing_single_neighbourhood(self, tmp_path):
        """Should return False when specific neighbourhood height file missing."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            assert is_neighbourhood_heights_available(neighbourhood_name='Haga') is False

    def test_handles_special_characters_in_name(self, tmp_path):
        """Should handle neighbourhood names with spaces and special chars."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        (footprints_dir / 'Sodra_Kortedala_heights.gpkg').write_bytes(b'mock')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            assert is_neighbourhood_heights_available(
                neighbourhood_name='Sodra Kortedala'
            ) is True

    def test_checks_all_areas_from_json(self, tmp_path):
        """Should check all areas from areas.json when no name specified."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "Haga"}, "102": {"name": "Annedal"}}')
        
        # Create only one file
        (footprints_dir / 'Haga_heights.gpkg').write_bytes(b'mock')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                assert is_neighbourhood_heights_available() is False

    def test_returns_true_when_all_areas_present(self, tmp_path):
        """Should return True when all areas have height files."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "Haga"}, "102": {"name": "Annedal"}}')
        
        (footprints_dir / 'Haga_heights.gpkg').write_bytes(b'mock')
        (footprints_dir / 'Annedal_heights.gpkg').write_bytes(b'mock')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                assert is_neighbourhood_heights_available() is True

    def test_skips_excluded_neighbourhoods(self, tmp_path):
        """Should skip neighbourhoods in SKIP_NEIGHBOURHOODS list."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "Haga"}, "408": {"name": "SkippedArea"}}')
        
        # Only create Haga - SkippedArea should be skipped
        (footprints_dir / 'Haga_heights.gpkg').write_bytes(b'mock')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                with patch('gbgsynth.data_utils.SKIP_NEIGHBOURHOODS', ['SkippedArea']):
                    assert is_neighbourhood_heights_available() is True


class TestGetMissingNeighbourhoodHeights:
    """Tests for get_missing_neighbourhood_heights function."""

    def test_returns_all_missing(self, tmp_path):
        """Should return all neighbourhoods when none have height files."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "Haga"}, "102": {"name": "Annedal"}}')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                missing = get_missing_neighbourhood_heights()
        
        assert len(missing) == 2
        names = [m['name'] for m in missing]
        assert 'Haga' in names
        assert 'Annedal' in names

    def test_returns_only_missing(self, tmp_path):
        """Should return only neighbourhoods missing height files."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        (footprints_dir / 'Haga_heights.gpkg').write_bytes(b'mock')
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "Haga"}, "102": {"name": "Annedal"}}')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                missing = get_missing_neighbourhood_heights()
        
        assert len(missing) == 1
        assert missing[0]['name'] == 'Annedal'
        assert missing[0]['code'] == '102'

    def test_returns_empty_when_all_present(self, tmp_path):
        """Should return empty list when all heights present."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        (footprints_dir / 'Haga_heights.gpkg').write_bytes(b'mock')
        (footprints_dir / 'Annedal_heights.gpkg').write_bytes(b'mock')
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "Haga"}, "102": {"name": "Annedal"}}')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                missing = get_missing_neighbourhood_heights()
        
        assert missing == []

    def test_excludes_skip_neighbourhoods(self, tmp_path):
        """Should not include neighbourhoods in SKIP_NEIGHBOURHOODS."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"408": {"name": "SkippedOne"}, "706": {"name": "SkippedTwo"}}')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                with patch('gbgsynth.data_utils.SKIP_NEIGHBOURHOODS', ['SkippedOne', 'SkippedTwo']):
                    missing = get_missing_neighbourhood_heights()
        
        assert missing == []


class TestGenerateNeighbourhoodHeights:
    """Tests for generate_neighbourhood_heights function."""

    def test_raises_when_geopandas_not_installed(self, tmp_path):
        """Should raise ImportError when geopandas not installed."""
        with patch.dict('sys.modules', {'geopandas': None}):
            with pytest.raises(ImportError, match='geopandas is required'):
                generate_neighbourhood_heights(
                    neighbourhood_name='Haga',
                    output_dir=tmp_path
                )

    def test_raises_when_dtcc_not_installed(self, tmp_path):
        """Should raise ImportError when dtcc not installed."""
        mock_gpd = MagicMock()
        
        with patch.dict('sys.modules', {'geopandas': mock_gpd, 'dtcc': None}):
            with pytest.raises(ImportError, match='dtcc is required'):
                generate_neighbourhood_heights(
                    neighbourhood_name='Haga',
                    output_dir=tmp_path
                )

    def test_returns_false_when_shapefile_missing(self, tmp_path):
        """Should return False when shapefile not available."""
        mock_gpd = MagicMock()
        mock_dtcc = MagicMock()
        mock_shapely = MagicMock()
        
        with patch.dict('sys.modules', {
            'geopandas': mock_gpd,
            'dtcc': mock_dtcc,
            'shapely.geometry': mock_shapely
        }):
            with patch('gbgsynth.data_utils.get_shapefile_path', return_value=None):
                result = generate_neighbourhood_heights(
                    neighbourhood_name='Haga',
                    output_dir=tmp_path
                )
        
        assert result is False

    def test_skips_excluded_neighbourhoods(self, tmp_path):
        """Should skip neighbourhoods in SKIP_NEIGHBOURHOODS."""
        mock_gpd = MagicMock()
        mock_dtcc = MagicMock()
        mock_shapely = MagicMock()
        
        mock_gdf = MagicMock()
        mock_gpd.read_file.return_value.to_crs.return_value = mock_gdf
        
        shapefile_path = tmp_path / 'pri.shp'
        shapefile_path.touch()
        
        with patch.dict('sys.modules', {
            'geopandas': mock_gpd,
            'dtcc': mock_dtcc,
            'shapely.geometry': mock_shapely
        }):
            with patch('gbgsynth.data_utils.get_shapefile_path', return_value=shapefile_path):
                with patch('gbgsynth.data_utils.get_missing_neighbourhood_heights', return_value=[]):
                    result = generate_neighbourhood_heights(output_dir=tmp_path)
        
        assert result is True


class TestEnsureNeighbourhoodHeightsAvailable:
    """Tests for ensure_neighbourhood_heights_available function."""

    def test_returns_true_when_available(self, tmp_path):
        """Should return True when all heights already available."""
        with patch('gbgsynth.data_utils.is_neighbourhood_heights_available', return_value=True):
            result = ensure_neighbourhood_heights_available()
        
        assert result is True

    def test_generates_when_missing_and_auto_generate_true(self, tmp_path):
        """Should generate when missing and auto_generate is True."""
        with patch('gbgsynth.data_utils.is_neighbourhood_heights_available', return_value=False):
            with patch('gbgsynth.data_utils.is_footprints_available', return_value=True):
                with patch('gbgsynth.data_utils.generate_neighbourhood_heights', return_value=True) as mock_gen:
                    result = ensure_neighbourhood_heights_available(auto_generate=True)
        
        assert result is True
        mock_gen.assert_called_once()

    def test_returns_false_when_missing_and_auto_generate_false(self):
        """Should return False when missing and auto_generate is False."""
        with patch('gbgsynth.data_utils.is_neighbourhood_heights_available', return_value=False):
            result = ensure_neighbourhood_heights_available(auto_generate=False)
        
        assert result is False

    def test_returns_false_when_footprints_unavailable(self):
        """Should return False when footprints not available."""
        with patch('gbgsynth.data_utils.is_neighbourhood_heights_available', return_value=False):
            with patch('gbgsynth.data_utils.is_footprints_available', return_value=False):
                result = ensure_neighbourhood_heights_available(auto_generate=True)
        
        assert result is False

    def test_handles_import_error(self):
        """Should return False when required packages not installed."""
        with patch('gbgsynth.data_utils.is_neighbourhood_heights_available', return_value=False):
            with patch('gbgsynth.data_utils.is_footprints_available', return_value=True):
                with patch('gbgsynth.data_utils.generate_neighbourhood_heights') as mock_gen:
                    mock_gen.side_effect = ImportError("dtcc not available")
                    result = ensure_neighbourhood_heights_available(auto_generate=True)
        
        assert result is False


class TestGetNeighbourhoodHeightsPath:
    """Tests for get_neighbourhood_heights_path function."""

    def test_returns_path_when_exists(self, tmp_path):
        """Should return path when height file exists."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        height_file = footprints_dir / 'Haga_heights.gpkg'
        height_file.write_bytes(b'mock content')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            result = get_neighbourhood_heights_path('Haga')
        
        assert result == height_file

    def test_returns_none_when_missing(self, tmp_path):
        """Should return None when height file doesn't exist."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            result = get_neighbourhood_heights_path('Haga')
        
        assert result is None

    def test_handles_special_characters(self, tmp_path):
        """Should handle neighbourhood names with special characters."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        height_file = footprints_dir / 'Sodra_Kortedala_heights.gpkg'
        height_file.write_bytes(b'mock content')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            result = get_neighbourhood_heights_path('Sodra Kortedala')
        
        assert result == height_file


class TestGetNeighbourhoodHeightsDir:
    """Tests for get_neighbourhood_heights_dir function."""

    def test_returns_footprints_dir(self):
        """Should return the footprints directory path."""
        result = get_neighbourhood_heights_dir()
        assert result.name == 'footprints'
        assert result.parent.name == 'data'


class TestSkipNeighbourhoods:
    """Tests for SKIP_NEIGHBOURHOODS constant."""

    def test_is_a_list(self):
        """SKIP_NEIGHBOURHOODS should be a list."""
        assert isinstance(SKIP_NEIGHBOURHOODS, list)

    def test_skip_logic_works_with_populated_list(self, tmp_path):
        """Should properly skip neighbourhoods when list is populated."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "TestSkip"}}')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                with patch('gbgsynth.data_utils.SKIP_NEIGHBOURHOODS', ['TestSkip']):
                    missing = get_missing_neighbourhood_heights()
        
        assert missing == []

    def test_skip_logic_works_with_empty_list(self, tmp_path):
        """Should process all neighbourhoods when skip list is empty."""
        footprints_dir = tmp_path / 'footprints'
        footprints_dir.mkdir()
        
        areas_json = tmp_path / 'areas.json'
        areas_json.write_text('{"101": {"name": "Haga"}}')
        
        with patch('gbgsynth.data_utils._FOOTPRINTS_DIR', footprints_dir):
            with patch('gbgsynth.data_utils._AREAS_JSON', areas_json):
                with patch('gbgsynth.data_utils.SKIP_NEIGHBOURHOODS', []):
                    missing = get_missing_neighbourhood_heights()
        
        assert len(missing) == 1
        assert missing[0]['name'] == 'Haga'
