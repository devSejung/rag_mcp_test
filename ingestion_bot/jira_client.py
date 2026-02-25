from jira import JIRA
from typing import List, Dict, Any, Generator
from datetime import datetime
from shared.config import settings
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JiraIngestionClient:
    """
    Client for fetching Jira issues incrementally.
    Supports comments, full context, and robust error handling.
    """
    
    def __init__(self):
        self.jira = JIRA(
            server=settings.JIRA_URL,
            basic_auth=(settings.JIRA_USERNAME, settings.JIRA_API_TOKEN)
        )
        self.project_key = settings.JIRA_PROJECT_KEY

    def fetch_updated_issues(self, last_sync_time: datetime) -> Generator[Dict[str, Any], None, None]:
        """
        Fetch issues updated after last_sync_time.
        Yields processed issue data including comments.
        """
        # Format date for JQL: YYYY/MM/DD HH:mm
        jql_date = last_sync_time.strftime("%Y/%m/%d %H:%M")
        jql = f'project = "{self.project_key}" AND updated >= "{jql_date}" ORDER BY updated ASC'
        
        start_at = 0
        max_results = 50
        
        while True:
            logger.info(f"Fetching Jira issues with JQL: {jql} (StartAt: {start_at})")
            issues = self._fetch_issues_page(jql, start_at, max_results)
            
            if not issues:
                break
                
            for issue in issues:
                try:
                    yield self._process_issue(issue)
                except Exception as e:
                    logger.error(f"Failed to process issue {issue.key}: {e}")
                    continue
            
            start_at += len(issues)
            if len(issues) < max_results:
                break

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def _fetch_issues_page(self, jql: str, start_at: int, max_results: int):
        return self.jira.search_issues(
            jql, 
            startAt=start_at, 
            maxResults=max_results,
            fields="summary,description,updated,status,comment,creator,created,components,parent"
        )

    @retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(5))
    def fetch_all_active_keys(self) -> set:
        """Fetch all active issue keys in the project for deletion scrub."""
        jql = f'project = "{self.project_key}"'
        start_at = 0
        max_results = 100
        active_keys = set()
        
        while True:
            issues = self.jira.search_issues(jql, startAt=start_at, maxResults=max_results, fields="key")
            if not issues:
                break
            active_keys.update([issue.key for issue in issues])
            start_at += len(issues)
            if len(issues) < max_results:
                break
                
        return active_keys

    def _process_issue(self, issue) -> Dict[str, Any]:
        """Convert Jira Issue object to a clean dictionary."""
        # 1. Basic Fields
        components = [c.name for c in issue.fields.components] if hasattr(issue.fields, "components") and issue.fields.components else []
        parent_key = issue.fields.parent.key if hasattr(issue.fields, "parent") and issue.fields.parent else None

        data = {
            "key": issue.key,
            "url": f"{settings.JIRA_URL}/browse/{issue.key}",
            "title": issue.fields.summary,
            "status": issue.fields.status.name,
            "updated": issue.fields.updated,
            "content": issue.fields.description or "",
            "components": components,
            "parent": parent_key,
            "comments": []
        }
        
        # 2. Process Comments
        comments_list = []
        if hasattr(issue.fields, "comment") and issue.fields.comment.comments:
            for comment in issue.fields.comment.comments:
                author = comment.author.displayName if hasattr(comment, "author") else "Unknown"
                created = comment.created
                body = comment.body
                
                comments_list.append({
                    "author": author,
                    "created": created,
                    "body": body
                })
        
        data["comments"] = comments_list
        return data
