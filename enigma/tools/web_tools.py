"""
Web Tools - Internet search and webpage fetching.

Tools:
  - web_search: Search the internet
  - fetch_webpage: Get content from a URL
"""

import urllib.request
import urllib.parse
import json
import re
from typing import Dict, Any, Optional
from .tool_registry import Tool


class WebSearchTool(Tool):
    """
    Search the internet using DuckDuckGo (no API key needed).
    """
    
    name = "web_search"
    description = "Search the internet for information. Returns top results with titles and snippets."
    parameters = {
        "query": "The search query string",
        "num_results": "Number of results to return (default: 5)",
    }
    
    def execute(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Search using DuckDuckGo Lite (simpler, more reliable).
        No API key required.
        """
        try:
            # Use DuckDuckGo Lite - simpler HTML
            encoded_query = urllib.parse.quote(query)
            url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode('utf-8')
            
            # Parse results from lite version
            results = []
            
            # Find result links - lite version has simpler structure
            # Links are in <a class="result-link" href="...">
            link_pattern = r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(link_pattern, html, re.IGNORECASE)
            
            if not matches:
                # Try alternative pattern
                link_pattern = r'<a[^>]*href="(https?://[^"]+)"[^>]*>([^<]+)</a>'
                matches = re.findall(link_pattern, html)
                # Filter out DuckDuckGo internal links
                matches = [(url, title) for url, title in matches 
                          if 'duckduckgo.com' not in url and len(title) > 10]
            
            for link, title in matches[:num_results]:
                # Clean up title
                title = re.sub(r'<[^>]+>', '', title).strip()
                if title and link.startswith('http'):
                    results.append({
                        "title": title,
                        "url": link,
                        "snippet": ""  # Lite version doesn't have snippets
                    })
            
            # If still no results, provide helpful message
            if not results:
                return {
                    "success": True,
                    "query": query,
                    "num_results": 0,
                    "results": [],
                    "note": "No results found. Try different search terms."
                }
            
            return {
                "success": True,
                "query": query,
                "num_results": len(results),
                "results": results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class FetchWebpageTool(Tool):
    """
    Fetch and extract text content from a webpage.
    """
    
    name = "fetch_webpage"
    description = "Fetch a webpage and extract its text content. Good for reading articles or documentation."
    parameters = {
        "url": "The URL to fetch",
        "max_length": "Maximum characters to return (default: 5000)",
    }
    
    def execute(self, url: str, max_length: int = 5000, **kwargs) -> Dict[str, Any]:
        """Fetch webpage and extract text."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; EnigmaBot/1.0)"
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')
            
            # Extract text content (remove HTML tags)
            # Remove script and style elements
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', html)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "... [truncated]"
            
            # Try to extract title
            title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Unknown"
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "content_length": len(text),
                "content": text
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Test
    search = WebSearchTool()
    result = search.execute("python programming")
    print(json.dumps(result, indent=2))
