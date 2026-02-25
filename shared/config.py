from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Internal API Settings (OpenAI Compatible)
    INTERNAL_API_BASE_URL: str = "https://api.yourcompany.com/v1"
    INTERNAL_API_KEY: str = "your-api-key"
    
    # Model IDs
    EMBEDDING_MODEL: str = "baai-bge-m3"  # BAAI recommended
    RERANKER_MODEL: str = "baai-bge-reranker-v2-m3"
    GENERATION_MODEL: str = "qwen2.5-8b-instruct"
    
    # Vector DB (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    
    # Collection Names
    JIRA_COLLECTION: str = "jira_docs"
    CONFLUENCE_COLLECTION: str = "confluence_docs"
    
    # Jira Ingestion Settings
    JIRA_URL: str = "https://your-domain.atlassian.net"
    JIRA_USERNAME: str = "your-email@example.com"
    JIRA_API_TOKEN: str = "your-api-token"
    JIRA_PROJECT_KEY: str = "PROJ"
    
    # Filtering Settings
    JIRA_ALLOWED_COMPONENTS: str = "" # Comma separated, e.g. "Frontend,Backend"
    MIN_CONTENT_LENGTH: int = 200
    
    # Confluence Ingestion Settings
    CONFLUENCE_URL: str = "https://your-domain.atlassian.net/wiki"
    CONFLUENCE_USERNAME: str = "your-email@example.com"
    CONFLUENCE_API_TOKEN: str = "your-api-token"
    CONFLUENCE_SPACE_KEY: str = "SPACE"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
