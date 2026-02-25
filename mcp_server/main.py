from fastmcp import FastMCP
from mcp_server.services.rag_service import RAGService
from shared.repositories.vector_store import JiraRepository, ConfluenceRepository
import asyncio
from typing import Optional

# Initialize repositories and services
jira_repo = JiraRepository()
confluence_repo = ConfluenceRepository()

jira_service = RAGService(jira_repo)
confluence_service = RAGService(confluence_repo)

# Create MCP Server
mcp = FastMCP("Company-RAG-Server")

@mcp.tool()
async def search_jira(query: str, component: Optional[str] = None, top_k: int = 5) -> str:
    """
    Search for information in Jira issues and projects.
    Use this for tasks, bugs, and project management queries.
    Pass component (e.g. 'Frontend', 'Backend') to filter by specific Jira components.
    """
    results = await jira_service.retrieve_and_rerank(query, top_k=top_k, filter_component=component)
    
    if not results:
        return "Jira에서 관련 정보를 찾지 못했습니다."
    
    formatted_results = []
    for res in results:
        payload = res["payload"]
        score = res.get("relevance_score", 0.0)
        formatted_results.append(
            f"--- Issue: {payload.get('key', 'N/A')} ---\n"
            f"Title: {payload.get('title', 'N/A')}\n"
            f"Status: {payload.get('status', 'N/A')}\n"
            f"Components: {', '.join(payload.get('components', []))}\n"
            f"Parent: {payload.get('parent', 'N/A')}\n"
            f"Content:\n{payload.get('full_formatted_text', payload.get('content', 'N/A'))}\n"
            f"Relevance Score: {score:.4f}\n"
        )
    
    return "\n".join(formatted_results)

@mcp.tool()
async def search_confluence(query: str, component: Optional[str] = None, top_k: int = 5) -> str:
    """
    Search for information in Confluence pages and wiki.
    Use this for documentation, guides, and meeting notes.
    Pass component (e.g. 'SPACE') to filter by specific Confluence space.
    """
    results = await confluence_service.retrieve_and_rerank(query, top_k=top_k, filter_component=component)
    
    if not results:
        return "Confluence에서 관련 정보를 찾지 못했습니다."
    
    formatted_results = []
    for res in results:
        payload = res["payload"]
        score = res.get("relevance_score", 0.0)
        formatted_results.append(
            f"--- Page: {payload.get('key', 'N/A')} ---\n"
            f"Title: {payload.get('title', 'N/A')}\n"
            f"URL: {payload.get('url', 'N/A')}\n"
            f"Space/Component: {', '.join(payload.get('components', []))}\n"
            f"Author: {payload.get('author', 'N/A')}\n"
            f"Content:\n{payload.get('full_formatted_text', payload.get('content', 'N/A'))}\n"
            f"Relevance Score: {score:.4f}\n"
        )
    
    return "\n".join(formatted_results)

if __name__ == "__main__":
    mcp.run()
