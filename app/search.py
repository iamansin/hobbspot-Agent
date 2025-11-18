"""Search service using DuckDuckGo for web search capabilities."""

import httpx
from typing import List, Dict, Optional
from loguru import logger
from app.utils import retry_with_backoff


class SearchService:
    """
    Manages web search operations using DuckDuckGo HTML search.
    
    Provides async web search with result formatting optimized for LLM
    consumption. Includes error handling, timeout management, and
    comprehensive logging.
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 13.8, 11.5
    """
    
    def __init__(self, api_key: str = "", timeout: float = 10.0):
        """
        Initialize search service.
        
        Args:
            api_key: Not used for DuckDuckGo (kept for compatibility)
            timeout: Request timeout in seconds (default: 10.0)
        """
        self.api_key = api_key  # Not used but kept for compatibility
        self.base_url = "https://html.duckduckgo.com/html/"
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            follow_redirects=True
        )
        logger.info(f"SearchService initialized with DuckDuckGo: timeout={timeout}s")
    
    async def search(self, query: str, count: int = 5) -> List[Dict[str, str]]:
        """
        Perform web search via DuckDuckGo.
        
        Executes a web search query and returns formatted results suitable
        for LLM consumption. Includes retry logic for transient failures.
        
        Args:
            query: Search query string
            count: Number of results to return (default: 5)
            
        Returns:
            List of formatted search results, each containing:
            - title: Result title
            - url: Result URL
            - description: Result snippet/description
            
        Raises:
            httpx.HTTPError: If the API request fails after retries
            
        Requirements: 7.1, 7.2, 7.3, 13.8
        """
        logger.info(f"Performing DuckDuckGo search: query='{query}', count={count}")
        
        try:
            # Use retry logic for API call
            results = await retry_with_backoff(
                self._perform_search,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                query=query,
                count=count
            )
            
            logger.info(f"Search completed: query='{query}', results_count={len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query='{query}': {e}")
            # Return empty results on failure rather than raising
            # This allows the LLM to continue without search results
            return []
    
    async def _perform_search(self, query: str, count: int) -> List[Dict[str, str]]:
        """
        Internal method to perform the actual API request.
        
        Args:
            query: Search query string
            count: Number of results to return
            
        Returns:
            List of formatted search results
            
        Raises:
            httpx.HTTPError: If the API request fails
        """
        # DuckDuckGo HTML search uses POST with form data
        data = {
            "q": query,
            "b": "",  # Start from first result
            "kl": "wt-wt"  # All regions
        }
        
        response = await self.client.post(
            self.base_url,
            data=data
        )
        
        response.raise_for_status()
        html_content = response.text
        
        return self._format_results(html_content, count)
    
    def _format_results(self, html_content: str, count: int) -> List[Dict[str, str]]:
        """
        Format DuckDuckGo HTML results for LLM consumption.
        
        Extracts relevant fields from DuckDuckGo HTML response and formats them
        into a clean structure suitable for LLM processing.
        
        Args:
            html_content: Raw HTML response from DuckDuckGo
            count: Maximum number of results to return
            
        Returns:
            List of formatted results with title, url, and description
            
        Requirements: 7.3
        """
        from html.parser import HTMLParser
        
        class DuckDuckGoParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current_result = {}
                self.in_result = False
                self.in_title = False
                self.in_snippet = False
                self.capture_data = False
                
            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                
                # Result container
                if tag == "div" and attrs_dict.get("class") == "result":
                    self.in_result = True
                    self.current_result = {}
                
                # Title link
                if self.in_result and tag == "a" and "result__a" in attrs_dict.get("class", ""):
                    self.in_title = True
                    self.capture_data = True
                    self.current_result["url"] = attrs_dict.get("href", "")
                
                # Snippet
                if self.in_result and tag == "a" and "result__snippet" in attrs_dict.get("class", ""):
                    self.in_snippet = True
                    self.capture_data = True
            
            def handle_endtag(self, tag):
                if tag == "a" and self.in_title:
                    self.in_title = False
                    self.capture_data = False
                
                if tag == "a" and self.in_snippet:
                    self.in_snippet = False
                    self.capture_data = False
                
                if tag == "div" and self.in_result:
                    if self.current_result.get("title") and self.current_result.get("url"):
                        self.results.append(self.current_result)
                    self.in_result = False
                    self.current_result = {}
            
            def handle_data(self, data):
                if self.capture_data:
                    data = data.strip()
                    if data:
                        if self.in_title:
                            self.current_result["title"] = data
                        elif self.in_snippet:
                            self.current_result["description"] = data
        
        parser = DuckDuckGoParser()
        try:
            parser.feed(html_content)
        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo HTML: {e}")
            return []
        
        # Limit results to requested count
        results = parser.results[:count]
        
        # Ensure all results have required fields
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", "")
            })
        
        return formatted_results
    
    async def close(self) -> None:
        """
        Close HTTP client and cleanup resources.
        
        Should be called during application shutdown to properly close
        the httpx client and release resources.
        
        Requirements: 11.5
        """
        await self.client.aclose()
        logger.info("SearchService closed")

