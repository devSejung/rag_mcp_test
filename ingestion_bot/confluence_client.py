from atlassian import Confluence
from typing import List, Dict, Any, Generator
from datetime import datetime
from shared.config import settings
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfluenceIngestionClient:
    """Client for fetching Confluence pages incrementally."""
    
    def __init__(self):
        self.confluence = Confluence(
            url=settings.CONFLUENCE_URL,
            username=settings.CONFLUENCE_USERNAME,
            password=settings.CONFLUENCE_API_TOKEN,
            cloud=True
        )
        self.space_key = settings.CONFLUENCE_SPACE_KEY

    def fetch_updated_pages(self, last_sync_time: datetime) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch pages updated after last_sync_time.
        Requires utilizing CQL (Confluence Query Language).
        """
        cql_date = last_sync_time.strftime("%Y-%m-%d %H:%M")
        # Format for CQL: lastmodified >= "YYYY-MM-DD HH:mm" AND order by asc
        # Note: Confluence CQL order by is somewhat limited, but we try to enforce chronological
        cql = f'space = "{self.space_key}" AND lastmodified >= "{cql_date}" order by lastmodified asc'
        
        start = 0
        limit = 50
        
        while True:
            logger.info(f"Fetching Confluence pages with CQL: {cql} (Start: {start})")
            
            # Fetch using advanced search (CQL)
            response = self._fetch_cql_page(cql, start, limit, expand='content.body.storage,content.history,content.version')
            
            results = response.get("results", [])
            if not results:
                break
                
            for page in results:
                try:
                    # cql returns content objects inside 'content' key
                    content_obj = page.get("content", page)
                    yield self._process_page(content_obj)
                except Exception as e:
                    logger.error(f"Failed to process page {page.get('id')}: {e}")
                    continue
            
            start += len(results)
            if start >= response.get("totalSize", start + 1) or len(results) < limit:
                break

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def _fetch_cql_page(self, cql: str, start: int, limit: int, expand: str = None):
        return self.confluence.cql(cql, start=start, limit=limit, expand=expand)

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def fetch_all_active_keys(self) -> set:
        """Fetch all active page keys in the space for deletion scrub."""
        cql = f'space = "{self.space_key}"'
        start = 0
        limit = 100
        active_keys = set()
        
        while True:
            response = self._fetch_cql_page(cql, start=start, limit=limit, expand="")
            results = response.get("results", [])
            if not results:
                break
                
            for page in results:
                page_id = page.get("content", page).get("id")
                if page_id:
                    active_keys.add(f"CONF-{page_id}")
                    
            start += len(results)
            if start >= response.get("totalSize", start + 1) or len(results) < limit:
                break
                
        return active_keys

    def _process_page(self, page) -> Dict[str, Any]:
        """Convert Confluence Page object to a clean dictionary."""
        page_id = page.get("id")
        title = page.get("title")
        url = f'{settings.CONFLUENCE_URL}{page.get("_links", {}).get("webui", "")}'
        
        # Confluence body is typically HTML
        body_html = page.get("body", {}).get("storage", {}).get("value", "")
        
        # Determine updated time (try to get from history or version)
        version_info = page.get("version", {})
        history_info = page.get("history", {})
        
        updated_raw = version_info.get("when") or history_info.get("createdDate")
        if not updated_raw:
             updated_raw = datetime.now().isoformat()
             
        author = version_info.get("by", {}).get("displayName", "Unknown")

        data = {
            "key": f"CONF-{page_id}", # Unified key structure
            "url": url,
            "title": title,
            "updated": updated_raw,
            "author": author,
            "content": body_html,
            "components": [self.space_key], # Wiki spaces act as major components
            "parent": None # Could be expanded to fetch page ancestors
        }
        
        return data
