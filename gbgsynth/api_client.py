"""
PxWeb API Client for Gothenburg City Statistics.

This module provides a robust client for interacting with the Gothenburg
PxWeb API, including metadata discovery and structured querying.

All API responses are cached locally to minimize API calls during development
and testing.
"""

import requests
import pandas as pd
import logging
import time
import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from gbgsynth.exceptions import APIError

logger = logging.getLogger(__name__)


class APICache:
    """
    File-based cache for API responses.
    
    Stores responses as JSON files in a local directory to avoid
    repeated API calls and reduce server load.
    """
    
    def __init__(
        self,
        cache_dir: str = ".gbgsynth_cache",
        enabled: bool = True,
        ttl_days: int = 7
    ):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cached responses
            enabled: Whether caching is enabled
            ttl_days: Time-to-live for cache entries in days
        """
        self.enabled = enabled
        self.ttl_days = ttl_days
        self.cache_dir = Path(cache_dir)
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._index_file = self.cache_dir / "cache_index.json"
            self._load_index()
    
    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        with open(self._index_file, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, indent=2)
    
    def _make_key(self, method: str, url: str, payload: Optional[Dict] = None) -> str:
        """Generate a unique cache key for a request."""
        key_data = f"{method}:{url}"
        if payload:
            key_data += f":{json.dumps(payload, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _is_expired(self, timestamp: str) -> bool:
        """Check if a cache entry has expired."""
        cached_time = datetime.fromisoformat(timestamp)
        return datetime.now() - cached_time > timedelta(days=self.ttl_days)
    
    def get(self, method: str, url: str, payload: Optional[Dict] = None) -> Optional[Dict]:
        """
        Retrieve a cached response.
        
        Args:
            method: HTTP method (GET/POST)
            url: Request URL
            payload: Request payload (for POST)
            
        Returns:
            Cached response data or None if not found/expired
        """
        if not self.enabled:
            return None
        
        key = self._make_key(method, url, payload)
        
        if key not in self._index:
            return None
        
        entry = self._index[key]
        
        # Check expiration
        if self._is_expired(entry['timestamp']):
            logger.debug(f"Cache expired for {key}")
            self.delete(key)
            return None
        
        # Load cached data
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Cache hit for {entry.get('description', key)}")
            return data
        except (json.JSONDecodeError, IOError):
            return None
    
    def set(
        self,
        method: str,
        url: str,
        data: Dict,
        payload: Optional[Dict] = None,
        description: str = ""
    ) -> None:
        """
        Store a response in the cache.
        
        Args:
            method: HTTP method
            url: Request URL
            data: Response data to cache
            payload: Request payload (for POST)
            description: Human-readable description
        """
        if not self.enabled:
            return
        
        key = self._make_key(method, url, payload)
        cache_file = self.cache_dir / f"{key}.json"
        
        # Save data
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        
        # Update index
        self._index[key] = {
            'timestamp': datetime.now().isoformat(),
            'url': url,
            'method': method,
            'description': description
        }
        self._save_index()
        
        logger.debug(f"Cached response for {description or key}")
    
    def delete(self, key: str) -> None:
        """Delete a cache entry."""
        if key in self._index:
            del self._index[key]
            self._save_index()
        
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
    
    def clear(self) -> None:
        """Clear all cached data."""
        if not self.enabled:
            return
        
        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        
        self._index = {}
        self._save_index()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        total_size = sum(
            f.stat().st_size for f in self.cache_dir.glob("*.json")
        )
        
        return {
            'enabled': True,
            'entries': len(self._index),
            'size_mb': round(total_size / 1024 / 1024, 2),
            'cache_dir': str(self.cache_dir),
            'ttl_days': self.ttl_days
        }


class PxWebClient:
    """
    Client for the Gothenburg City PxWeb API.
    
    Features:
    - Automatic caching of all API responses (default: 7 days TTL)
    - Rate limiting to be respectful to the API
    - Retry logic for transient failures
    
    All responses are cached to `.gbgsynth_cache/` by default, so repeated
    runs don't make unnecessary API calls.
    """

    BASE_URL = "https://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/"

    def __init__(
        self,
        request_delay: float = 0.2,
        max_retries: int = 3,
        cache_enabled: bool = True,
        cache_dir: str = ".gbgsynth_cache",
        cache_ttl_days: int = 7
    ):
        """
        Initialize the PxWeb client.
        
        Args:
            request_delay: Delay between requests in seconds (default: 0.2)
            max_retries: Maximum retries for failed requests (default: 3)
            cache_enabled: Whether to cache API responses locally (default: True)
            cache_dir: Directory for cached responses (default: .gbgsynth_cache)
            cache_ttl_days: Days until cached responses expire (default: 7)
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GbgSynth/0.3.0 Python Population Synthesizer",
            "Accept": "application/json"
        })
        self.request_delay = request_delay
        self.max_retries = max_retries
        self._last_request_time = 0
        
        # Initialize cache
        self.cache = APICache(
            cache_dir=cache_dir,
            enabled=cache_enabled,
            ttl_days=cache_ttl_days
        )
        
        if cache_enabled:
            logger.debug(f"API cache enabled: {cache_dir} (TTL: {cache_ttl_days} days)")

    def _rate_limited_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make a rate-limited request with retry logic."""
        # Enforce delay between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        
        for attempt in range(self.max_retries):
            self._last_request_time = time.time()
            
            if method == 'GET':
                response = self.session.get(url, **kwargs)
            else:
                response = self.session.post(url, **kwargs)
            
            if response.status_code == 429:  # Rate limited
                wait_time = (attempt + 1) * 3 + 1  # More aggressive backoff: 4s, 7s, 10s
                logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            
            return response
        
        return response  # Return last response even if still 429

    def fetch_metadata(self, table_path: str) -> Dict[str, Any]:
        """
        Fetch metadata for a specific table to discover available variables,
        codes, and values.

        Args:
            table_path: Relative path to the table (e.g., "Befolkning/.../63_FolkmHHtypPRI.px")

        Returns:
            Dictionary containing table metadata including all variables and their options

        Raises:
            APIError: If the API request fails
        """
        url = f"{self.BASE_URL}{table_path}"
        
        # Check cache first
        cached = self.cache.get('GET', url)
        if cached is not None:
            return cached
        
        try:
            response = self._rate_limited_request('GET', url, timeout=30)
            response.raise_for_status()
            metadata = response.json()
            
            # Cache the response
            self.cache.set('GET', url, metadata, description=f"metadata:{table_path}")
            
            logger.debug(f"Fetched metadata for {table_path}")
            return metadata
            
        except requests.RequestException as e:
            raise APIError(f"Failed to fetch metadata: {e}", url=url)

    def get_area_codes(self, sample_table: str = "Befolkning/Folkmängd/Folkmängd helår/63_FolkmHHtypPRI.px") -> Dict[str, Dict[str, str]]:
        """
        Dynamically discovers all available Primary Area (Primärområden) codes
        and names from the API metadata.

        Args:
            sample_table: Any table that contains the 'Område' variable

        Returns:
            Dictionary mapping area codes to area info:
            {"107": {"name": "107 Haga", "api_value": "107 Haga"}}

        Example:
            >>> client = PxWebClient()
            >>> areas = client.get_area_codes()
            >>> print(areas['107']['name'])  # '107 Haga'
        """
        metadata = self.fetch_metadata(sample_table)

        # Find the 'Område' variable in metadata
        for var in metadata.get('variables', []):
            if var.get('text') == 'Område':
                codes = var.get('values', [])
                names = var.get('valueTexts', [])
                
                # Map short code to full API value and display name
                area_dict = {}
                for api_value, full_name in zip(codes, names):
                    # Extract short code (e.g., "107" from "107 Haga")
                    short_code = api_value.split()[0] if ' ' in api_value else api_value
                    area_dict[short_code] = {
                        'name': full_name,
                        'api_value': api_value  # This is what the API expects
                    }
                
                logger.info(f"Discovered {len(area_dict)} primary areas")
                return area_dict

        logger.warning("No 'Område' variable found in metadata")
        return {}

    def query_table(
        self,
        table_path: str,
        area_code: str,
        year: int,
        additional_filters: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Executes a POST query to fetch specific census data.

        Args:
            table_path: Relative path to the table
            area_code: Primary area code (e.g., "107")
            year: Year to query (e.g., 2024)
            additional_filters: Optional dict of variable codes to values
                               (e.g., {"Ålder": ["0-17 år", "18-64 år"]})

        Returns:
            Pandas DataFrame with query results

        Example:
            >>> client = PxWebClient()
            >>> df = client.query_table(
            ...     "Övrigt/Personbilar/10_Bilar_PRI.px",
            ...     area_code="107",
            ...     year=2024
            ... )
        """
        url = f"{self.BASE_URL}{table_path}"

        # Build query payload
        query_payload = {
            "query": [
                {
                    "code": "Område",
                    "selection": {
                        "filter": "item",
                        "values": [area_code]
                    }
                },
                {
                    "code": "År",
                    "selection": {
                        "filter": "item",
                        "values": [str(year)]
                    }
                }
            ],
            "response": {"format": "json"}
        }

        # Add additional variable selections if provided
        if additional_filters:
            for code, values in additional_filters.items():
                query_payload["query"].append({
                    "code": code,
                    "selection": {
                        "filter": "item",
                        "values": values
                    }
                })

        # Check cache first
        cached = self.cache.get('POST', url, query_payload)
        if cached is not None:
            df = self._parse_json_response(cached)
            return df

        try:
            response = self._rate_limited_request('POST', url, json=query_payload, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            
            # Cache the response
            cache_desc = f"query:{table_path}:{area_code}:{year}"
            self.cache.set('POST', url, json_data, query_payload, description=cache_desc)
            
            df = self._parse_json_response(json_data)
            logger.debug(f"Queried {table_path} for area {area_code}, year {year}")
            
            # Check for "Sekretess" (privacy suppression)
            if '..' in df.values.astype(str):
                logger.warning(f"Privacy suppression detected in {table_path} for area {area_code}")
            
            return df
            
        except requests.RequestException as e:
            raise APIError(f"Failed to query {table_path}: {e}", url=url)

    def query_all_variables(
        self,
        table_path: str,
        area_code: str,
        year: int
    ) -> pd.DataFrame:
        """
        Query a table with all possible variable values (no filtering).

        Args:
            table_path: Relative path to the table
            area_code: Primary area code
            year: Year to query

        Returns:
            Pandas DataFrame with all combinations
        """
        url = f"{self.BASE_URL}{table_path}"

        query_payload = {
            "query": [
                {
                    "code": "Område",
                    "selection": {
                        "filter": "item",
                        "values": [area_code]
                    }
                },
                {
                    "code": "År",
                    "selection": {
                        "filter": "item",
                        "values": [str(year)]
                    }
                }
            ],
            "response": {"format": "json"}
        }

        # Get metadata to find all other variables
        metadata = self.fetch_metadata(table_path)
        
        for var in metadata.get('variables', []):
            var_code = var.get('code')
            # Skip the variables we've already specified
            if var_code not in ['Område', 'År']:
                # Select all values for this variable
                query_payload["query"].append({
                    "code": var_code,
                    "selection": {
                        "filter": "all",
                        "values": ["*"]
                    }
                })

        # Check cache first
        cached = self.cache.get('POST', url, query_payload)
        if cached is not None:
            return self._parse_json_response(cached)

        try:
            response = self._rate_limited_request('POST', url, json=query_payload, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            
            # Cache the response
            cache_desc = f"query_all:{table_path}:{area_code}:{year}"
            self.cache.set('POST', url, json_data, query_payload, description=cache_desc)
            
            return self._parse_json_response(json_data)
        except requests.RequestException as e:
            raise APIError(f"Failed to query {table_path}: {e}", url=url)

    def _parse_json_response(self, json_data: Dict) -> pd.DataFrame:
        """
        Converts the complex PxWeb JSON response into a flat, readable Pandas DataFrame.

        Args:
            json_data: Raw JSON response from PxWeb API

        Returns:
            Flattened DataFrame with columns for each variable and a value column
        """
        columns = [col['text'] for col in json_data.get('columns', [])]
        data_rows = []

        for item in json_data.get('data', []):
            # Combine keys (dimension values) with values (counts)
            row = item.get('key', []) + item.get('values', [])
            data_rows.append(row)

        df = pd.DataFrame(data_rows, columns=columns)
        
        # Convert the last column (typically the count) to numeric
        if len(df.columns) > 0:
            last_col = df.columns[-1]
            df[last_col] = pd.to_numeric(df[last_col], errors='coerce')
        
        # Workaround: PxWeb API sometimes returns 'NoContent' as the value column name
        # Rename it to 'Antal' (Swedish for "count") for consistency
        if 'NoContent' in df.columns and 'Antal' not in df.columns:
            df = df.rename(columns={'NoContent': 'Antal'})
        
        return df

    def clear_cache(self) -> None:
        """Clear all cached API responses."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache info (entries, size, etc.)
        """
        return self.cache.get_stats()
