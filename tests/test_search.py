"""Unit tests for search service."""

import pytest
import asyncio
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from app.search import SearchService


@pytest.fixture
def search_service():
    """Create a SearchService instance for testing."""
    service = SearchService(api_key="test_api_key", timeout=10.0)
    yield service
    # Cleanup
    asyncio.run(service.close())


@pytest.fixture
def mock_brave_response():
    """Create a mock Brave API response."""
    return {
        "web": {
            "results": [
                {
                    "title": "Python Programming Guide",
                    "url": "https://example.com/python-guide",
                    "description": "A comprehensive guide to Python programming"
                },
                {
                    "title": "Learn Python in 2024",
                    "url": "https://example.com/learn-python",
                    "description": "Modern Python tutorial for beginners"
                },
                {
                    "title": "Python Best Practices",
                    "url": "https://example.com/best-practices",
                    "description": "Industry-standard Python coding practices"
                }
            ]
        }
    }


@pytest.fixture
def empty_brave_response():
    """Create an empty Brave API response."""
    return {
        "web": {
            "results": []
        }
    }


class TestSearchService:
    """Tests for SearchService class."""
    
    def test_initialization(self):
        """Test search service initialization."""
        service = SearchService(api_key="test_key", timeout=15.0)
        
        assert service.api_key == "test_key"
        assert service.timeout == 15.0
        assert service.base_url == "https://api.search.brave.com/res/v1/web/search"
        assert service.client is not None
        
        # Cleanup
        asyncio.run(service.close())
    
    def test_initialization_default_timeout(self):
        """Test search service initialization with default timeout."""
        service = SearchService(api_key="test_key")
        
        assert service.timeout == 10.0
        
        # Cleanup
        asyncio.run(service.close())
    
    @pytest.mark.asyncio
    async def test_search_success(self, search_service, mock_brave_response):
        """Test successful search with mocked Brave API response."""
        # Mock the httpx client get method
        mock_response = MagicMock()
        mock_response.json.return_value = mock_brave_response
        mock_response.raise_for_status = MagicMock()
        
        search_service.client.get = AsyncMock(return_value=mock_response)
        
        # Perform search
        results = await search_service.search("Python programming", count=3)
        
        # Verify results
        assert len(results) == 3
        assert results[0]["title"] == "Python Programming Guide"
        assert results[0]["url"] == "https://example.com/python-guide"
        assert results[0]["description"] == "A comprehensive guide to Python programming"
        assert results[1]["title"] == "Learn Python in 2024"
        assert results[2]["title"] == "Python Best Practices"
    
    @pytest.mark.asyncio
    async def test_search_with_custom_count(self, search_service, mock_brave_response):
        """Test search with custom result count."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_brave_response
        mock_response.raise_for_status = MagicMock()
        
        search_service.client.get = AsyncMock(return_value=mock_response)
        
        # Perform search with custom count
        results = await search_service.search("test query", count=10)
        
        # Verify the API was called with correct parameters
        search_service.client.get.assert_called_once()
        call_args = search_service.client.get.call_args
        assert call_args.kwargs["params"]["count"] == 10
        assert call_args.kwargs["params"]["q"] == "test query"
    
    @pytest.mark.asyncio
    async def test_search_empty_results(self, search_service, empty_brave_response):
        """Test search with empty results."""
        mock_response = MagicMock()
        mock_response.json.return_value = empty_brave_response
        mock_response.raise_for_status = MagicMock()
        
        search_service.client.get = AsyncMock(return_value=mock_response)
        
        # Perform search
        results = await search_service.search("nonexistent query")
        
        # Verify empty results
        assert len(results) == 0
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_api_failure(self, search_service):
        """Test search handles API failures gracefully."""
        # Mock API failure
        search_service.client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "API Error",
                request=MagicMock(),
                response=MagicMock(status_code=500)
            )
        )
        
        # Perform search - should return empty list instead of raising
        results = await search_service.search("test query")
        
        # Verify empty results on failure
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_timeout(self, search_service):
        """Test search handles timeout scenarios."""
        # Mock timeout
        search_service.client.get = AsyncMock(
            side_effect=httpx.TimeoutException("Request timeout")
        )
        
        # Perform search - should return empty list instead of raising
        results = await search_service.search("test query")
        
        # Verify empty results on timeout
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_network_error(self, search_service):
        """Test search handles network errors."""
        # Mock network error
        search_service.client.get = AsyncMock(
            side_effect=httpx.NetworkError("Network unreachable")
        )
        
        # Perform search - should return empty list instead of raising
        results = await search_service.search("test query")
        
        # Verify empty results on network error
        assert results == []
    
    def test_format_results(self, search_service, mock_brave_response):
        """Test result formatting."""
        formatted = search_service._format_results(mock_brave_response)
        
        assert len(formatted) == 3
        assert all("title" in result for result in formatted)
        assert all("url" in result for result in formatted)
        assert all("description" in result for result in formatted)
        
        # Verify specific formatting
        assert formatted[0]["title"] == "Python Programming Guide"
        assert formatted[1]["url"] == "https://example.com/learn-python"
        assert formatted[2]["description"] == "Industry-standard Python coding practices"
    
    def test_format_results_missing_fields(self, search_service):
        """Test result formatting with missing fields."""
        incomplete_response = {
            "web": {
                "results": [
                    {
                        "title": "Test Title"
                        # Missing url and description
                    },
                    {
                        "url": "https://example.com/test"
                        # Missing title and description
                    }
                ]
            }
        }
        
        formatted = search_service._format_results(incomplete_response)
        
        assert len(formatted) == 2
        assert formatted[0]["title"] == "Test Title"
        assert formatted[0]["url"] == ""
        assert formatted[0]["description"] == ""
        assert formatted[1]["title"] == ""
        assert formatted[1]["url"] == "https://example.com/test"
    
    def test_format_results_empty_response(self, search_service, empty_brave_response):
        """Test result formatting with empty response."""
        formatted = search_service._format_results(empty_brave_response)
        
        assert formatted == []
    
    def test_format_results_malformed_response(self, search_service):
        """Test result formatting with malformed response."""
        malformed_response = {
            "web": {}  # Missing 'results' key
        }
        
        formatted = search_service._format_results(malformed_response)
        
        assert formatted == []
    
    def test_format_results_no_web_key(self, search_service):
        """Test result formatting when 'web' key is missing."""
        no_web_response = {
            "other_data": "value"
        }
        
        formatted = search_service._format_results(no_web_response)
        
        assert formatted == []
    
    @pytest.mark.asyncio
    async def test_search_request_headers(self, search_service, mock_brave_response):
        """Test that search includes correct headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_brave_response
        mock_response.raise_for_status = MagicMock()
        
        search_service.client.get = AsyncMock(return_value=mock_response)
        
        # Perform search
        await search_service.search("test query")
        
        # Verify headers were included
        call_args = search_service.client.get.call_args
        headers = call_args.kwargs["headers"]
        
        assert headers["Accept"] == "application/json"
        assert headers["Accept-Encoding"] == "gzip"
        assert headers["X-Subscription-Token"] == "test_api_key"
    
    @pytest.mark.asyncio
    async def test_search_retry_logic(self, search_service, mock_brave_response):
        """Test that search retries on failure."""
        # Mock first two calls to fail, third to succeed
        mock_response = MagicMock()
        mock_response.json.return_value = mock_brave_response
        mock_response.raise_for_status = MagicMock()
        
        call_count = 0
        
        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.HTTPStatusError(
                    "Temporary error",
                    request=MagicMock(),
                    response=MagicMock(status_code=503)
                )
            return mock_response
        
        search_service.client.get = mock_get
        
        # Perform search - should succeed after retries
        results = await search_service.search("test query")
        
        # Verify retries occurred
        assert call_count == 3
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_close(self, search_service):
        """Test closing the search service."""
        # Mock the aclose method
        search_service.client.aclose = AsyncMock()
        
        # Close the service
        await search_service.close()
        
        # Verify aclose was called
        search_service.client.aclose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multiple_searches(self, search_service, mock_brave_response):
        """Test performing multiple searches."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_brave_response
        mock_response.raise_for_status = MagicMock()
        
        search_service.client.get = AsyncMock(return_value=mock_response)
        
        # Perform multiple searches
        results1 = await search_service.search("query 1")
        results2 = await search_service.search("query 2")
        results3 = await search_service.search("query 3")
        
        # Verify all searches succeeded
        assert len(results1) == 3
        assert len(results2) == 3
        assert len(results3) == 3
        
        # Verify client was called 3 times
        assert search_service.client.get.call_count == 3

