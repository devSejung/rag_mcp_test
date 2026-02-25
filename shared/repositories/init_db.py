import asyncio
import logging
from shared.config import settings
from shared.repositories.vector_store import JiraRepository, ConfluenceRepository

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing Qdrant Collections for Hybrid Search...")
    
    jira_repo = JiraRepository()
    jira_repo.initialize_collection()
    
    conf_repo = ConfluenceRepository()
    conf_repo.initialize_collection()
    
    print("Successfully recreated collections with Dense and Sparse configurations.")

if __name__ == "__main__":
    main()
