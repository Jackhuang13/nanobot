"""Web tools: web_search and web_fetch."""

import abc
import html
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import ollama

from nanobot.agent.tools.base import Tool
from nanobot.config.schema import WebSearchConfig, WebFetchConfig

from loguru import logger

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks
PREVIEW_LIMIT = 300  # Character limit for search snippets and previews


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Search Providers
# ---------------------------------------------------------------------------

class SearchProvider(abc.ABC):
    """Abstract base class for web search providers."""
    
    @abc.abstractmethod
    async def search(self, query: str, count: int) -> str:
        """Execute search and return formatted string results."""
        pass


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def search(self, query: str, count: int) -> str:
        if not self.api_key:
            return "Error: BRAVE_API_KEY not configured"
        
        try:
            n = min(max(count, 1), 10)
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    timeout=10.0
                )
                r.raise_for_status()
            
            results = r.json().get("web", {}).get("results", [])
            if not results:
                return f"No results for: {query}"
            
            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    if len(desc) > PREVIEW_LIMIT:
                        desc = f"{desc[:PREVIEW_LIMIT]}..."
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error executing Brave search: {e}")
            return f"Error executing Brave search: {e}"


class OllamaSearchProvider(SearchProvider):
    """Ollama Native Web Search API provider."""
    
    def __init__(self, api_base: str, api_key: str = "", options: dict[str, Any] | None = None):
        self.api_base = api_base or "https://ollama.com"
        self.api_key = api_key
        self.options = options
        
        client_kwargs = {"host": self.api_base}
        # Only add headers if api_key is provided, as some backends might not expect it
        if self.api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
            
        self.client = ollama.Client(**client_kwargs)
        
    async def search(self, query: str, count: int) -> str:
        try:
            import asyncio
            
            response = await asyncio.to_thread(self.client.web_search, query=query)

            # Based on user spec: {'results': [{'title':..., 'url':..., 'content':...}]}
            results = response.get('results', [])
            if not results:
                return f"No results for: {query}"
            
            # Limit results
            results = results[:count]
            
            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item.get('title', 'No title')}\n   {item.get('url', 'No URL')}")
                if content := item.get("content"):
                    if len(content) > PREVIEW_LIMIT:
                        content = f"{content[:PREVIEW_LIMIT]}..."
                    lines.append(f"   {content}")
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error executing Ollama search: {e}")
            return f"Error executing Ollama search: {e}"


# ---------------------------------------------------------------------------
# Fetch Providers
# ---------------------------------------------------------------------------

class FetchProvider(abc.ABC):
    """Abstract base class for web fetch providers."""
    
    @abc.abstractmethod
    async def fetch(self, url: str, extract_mode: str, max_chars: int) -> str:
        """Execute fetch and return formatted string/JSON."""
        pass


class ReadabilityFetchProvider(FetchProvider):
    """Default provider using httpx + accessibility."""
    
    def __init__(self):
        pass

    async def fetch(self, url: str, extract_mode: str, max_chars: int) -> str:
        from readability import Document

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False)

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
            
            ctype = r.headers.get("content-type", "")
            
            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2, ensure_ascii=False), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extract_mode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"
            
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            
            return json.dumps({"url": url, "finalUrl": str(r.url), "status": r.status_code,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    def _to_markdown(self, html_content: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html_content, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))


class OllamaFetchProvider(FetchProvider):
    """Ollama Native Web Fetch API provider."""
    
    def __init__(self, api_base: str, api_key: str = "", options: dict[str, Any] | None = None):
        self.api_base = api_base or "https://ollama.com"
        self.api_key = api_key
        self.options = options
        
        client_kwargs = {"host": self.api_base}
        if self.api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}
            
        self.client = ollama.Client(**client_kwargs)

    async def fetch(self, url: str, extract_mode: str, max_chars: int) -> str:
        try:
             import asyncio
             
             result = await asyncio.to_thread(self.client.web_fetch, url=url)
             
             # Result object extraction
             title = getattr(result, 'title', '') or (result.get('title', '') if isinstance(result, dict) else '')
             content = getattr(result, 'content', '') or (result.get('content', '') if isinstance(result, dict) else '')
             links = getattr(result, 'links', []) or (result.get('links', []) if isinstance(result, dict) else [])
             
             text = f"# {title}\n\n{content}"
             
             truncated = len(text) > max_chars
             if truncated:
                 text = text[:max_chars]
                 
             return json.dumps({
                 "url": url, 
                 "extractor": "ollama", 
                 "truncated": truncated, 
                 "length": len(text), 
                 "text": text,
                 "links": links
             })
             
        except Exception as e:
            logger.error(f"Error executing Ollama fetch: {e}")
            return json.dumps({"error": f"Error executing Ollama fetch: {e}", "url": url})



# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class WebSearchTool(Tool):
    """Search the web using Brave Search or Ollama Native Search."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(self, config: WebSearchConfig | None = None, api_key: str | None = None):
        if config:
            self.config = config
        else:
            self.config = WebSearchConfig(
                provider="brave",
                api_key=api_key or os.environ.get("BRAVE_API_KEY", ""),
            )
        
        self.provider: SearchProvider
        if self.config.provider == "ollama":
            self.provider = OllamaSearchProvider(
                api_base=self.config.api_base,
                api_key=self.config.api_key,
                options=self.config.options
            )
        else:
            self.provider = BraveSearchProvider(api_key=self.config.api_key)

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if not self.config.enabled:
            return "Web search is disabled."

        return await self.provider.search(
            query=query, 
            count=count or self.config.max_results
        )


class WebFetchTool(Tool):
    """Fetch and extract web content using Readability or Ollama Native Fetch."""
    
    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }
    
    def __init__(self, config: WebFetchConfig | None = None, max_chars: int = 50000):
        if config:
            self.config = config
        else:
            self.config = WebFetchConfig(
                provider="readability",
            )
        self.max_chars = max_chars
        
        self.provider: FetchProvider
        if self.config.provider == "ollama":
            self.provider = OllamaFetchProvider(
                api_base=self.config.api_base,
                api_key=self.config.api_key,
                options=self.config.options
            )
        else:
            self.provider = ReadabilityFetchProvider()
    
    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        if not self.config.enabled:
             return "Web fetch is disabled."

        return await self.provider.fetch(
            url=url, 
            extract_mode=extractMode, 
            max_chars=maxChars or self.max_chars
        )


