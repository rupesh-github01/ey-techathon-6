import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import re
import time


def web_search(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo's HTML interface.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Dict with agent name, data list, and summary
    """
    print(f"[Web Agent] Searching for: {query}")
    
    try:
        # DuckDuckGo HTML search URL
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        
        # DuckDuckGo HTML results - try multiple selectors
        result_divs = soup.find_all('div', class_='result') or soup.find_all('div', class_='results_links')
        
        # Also try links with result__a class directly
        if not result_divs:
            result_links = soup.find_all('a', class_='result__a')
            for i, link in enumerate(result_links[:max_results]):
                title = link.get_text(strip=True)
                href = link.get('href', '')
                
                # Try to find associated snippet
                parent = link.find_parent('div')
                snippet_elem = parent.find('a', class_='result__snippet') if parent else None
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                if title and href:
                    # Decode DuckDuckGo redirect URL if present
                    if 'uddg=' in href:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        href = urllib.parse.unquote(parsed.get('uddg', [href])[0])
                    
                    results.append({
                        "rank": i + 1,
                        "title": title,
                        "url": href,
                        "display_url": href[:50] + "..." if len(href) > 50 else href,
                        "snippet": snippet,
                        "source": "DuckDuckGo"
                    })
        else:
            for i, result_div in enumerate(result_divs[:max_results]):
                try:
                    # Get the title and link
                    title_elem = result_div.find('a', class_='result__a')
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    url_elem = result_div.find('a', class_='result__url')
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        link = title_elem.get('href', '')
                        
                        # DuckDuckGo wraps URLs, need to extract actual URL
                        if 'uddg=' in link:
                            import urllib.parse
                            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                            link = urllib.parse.unquote(parsed.get('uddg', [link])[0])
                        
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        display_url = url_elem.get_text(strip=True) if url_elem else link
                        
                        results.append({
                            "rank": i + 1,
                            "title": title,
                            "url": link,
                            "display_url": display_url,
                            "snippet": snippet,
                            "source": "DuckDuckGo"
                        })
                except Exception as e:
                    print(f"[Web Agent] Error parsing result {i}: {e}")
                    continue
        
        # If still no results, try a different approach - look for all links with snippets
        if not results:
            all_links = soup.find_all('a', href=True)
            for i, link in enumerate(all_links):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Filter out navigation/internal links
                if (text and len(text) > 20 and 
                    not href.startswith('/') and 
                    not href.startswith('#') and
                    'duckduckgo.com' not in href and
                    href.startswith('http')):
                    
                    results.append({
                        "rank": len(results) + 1,
                        "title": text[:100],
                        "url": href,
                        "display_url": href[:50] + "..." if len(href) > 50 else href,
                        "snippet": "",
                        "source": "DuckDuckGo"
                    })
                    
                    if len(results) >= max_results:
                        break
        
        summary = f"Web Agent found {len(results)} search results for '{query}'"
        print(f"[Web Agent] {summary}")
        
        return {
            "agent": "web",
            "data": results,
            "summary": summary,
            "query": query
        }
        
        return {
            "agent": "web",
            "data": results,
            "summary": summary,
            "query": query
        }
        
    except requests.exceptions.Timeout:
        return {
            "agent": "web",
            "data": [],
            "summary": f"Web search timed out for '{query}'",
            "query": query
        }
    except requests.exceptions.RequestException as e:
        return {
            "agent": "web",
            "data": [],
            "summary": f"Web search failed: {str(e)}",
            "query": query
        }
    except Exception as e:
        return {
            "agent": "web",
            "data": [],
            "summary": f"Web search error: {str(e)}",
            "query": query
        }


def fetch_page_content(url: str, max_chars: int = 5000) -> Dict[str, Any]:
    """
    Fetch and extract main text content from a webpage.
    
    Args:
        url: URL to fetch
        max_chars: Maximum characters to return
        
    Returns:
        Dict with page title, content, and metadata
    """
    print(f"[Web Agent] Fetching page: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Get title
        title = soup.title.string if soup.title else "No title"
        
        # Get main content - try common content containers
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.post', '.article']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body if soup.body else soup
        
        # Extract text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Truncate if needed
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"
        
        return {
            "url": url,
            "title": title,
            "content": text,
            "char_count": len(text),
            "success": True
        }
        
    except Exception as e:
        return {
            "url": url,
            "title": "Error",
            "content": f"Failed to fetch page: {str(e)}",
            "char_count": 0,
            "success": False
        }


def pharma_web_search(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Search for pharmaceutical-related information on the web.
    Automatically appends relevant pharma terms to improve results.
    """
    # Add pharma context to generic queries
    pharma_terms = ['pharmaceutical', 'drug', 'FDA', 'clinical trial', 'medicine']
    
    # Check if query already has pharma context
    query_lower = query.lower()
    has_pharma_context = any(term.lower() in query_lower for term in pharma_terms)
    
    if not has_pharma_context:
        # Add pharmaceutical context
        enhanced_query = f"{query} pharmaceutical drug"
    else:
        enhanced_query = query
    
    return web_search(enhanced_query, max_results)


if __name__ == '__main__':
    # Test the web search
    import json
    
    print("Testing Web Agent...")
    print("=" * 50)
    
    # Test search
    query = input("Enter search query (default: 'diabetes drug market 2024'): ").strip()
    if not query:
        query = "diabetes drug market 2024"
    
    results = web_search(query)
    print(f"\nResults for '{query}':")
    print(json.dumps(results, indent=2, default=str)[:2000])
