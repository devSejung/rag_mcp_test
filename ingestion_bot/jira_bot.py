import time
import schedule
import logging
from datetime import datetime
import asyncio
from typing import List, Dict, Any

from shared.config import settings
from shared.apis.openai_client import ai_client
from shared.apis.sparse_embedder import sparse_embedder
from shared.repositories.vector_store import JiraRepository
from ingestion_bot.jira_client import JiraIngestionClient
from ingestion_bot.jira_processor import JiraProcessor
from ingestion_bot.state_manager import StateManager

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("jira_bot.log")
    ]
)
logger = logging.getLogger("JiraBot")

class JiraBot:
    def __init__(self):
        self.jira_client = JiraIngestionClient()
        allowed_comps = [c.strip() for c in settings.JIRA_ALLOWED_COMPONENTS.split(',')] if settings.JIRA_ALLOWED_COMPONENTS else []
        self.processor = JiraProcessor(allowed_components=allowed_comps, min_length=settings.MIN_CONTENT_LENGTH)
        self.vector_repo = JiraRepository()
        self.state_manager = StateManager()
        
    async def run_sync_cycle(self):
        """Execute one full sync cycle."""
        logger.info("Starting Sync Cycle...")
        
        # 1. Load Last Sync Time
        last_sync = self.state_manager.get_last_sync_time(default_days_ago=365) # Fetch all history initially
        logger.info(f"Last sync time: {last_sync}")
        
        new_sync_time = last_sync
        processed_count = 0
        
        try:
            # 2. Fetch Changed Issues
            for issue_data in self.jira_client.fetch_updated_issues(last_sync):
                try:
                    # Track successful processing time FIRST to avoid dropping items on skip/error
                    from dateutil.parser import parse
                    try:
                        issue_time = parse(issue_data['updated']).replace(tzinfo=None)
                        if issue_time > new_sync_time:
                            new_sync_time = issue_time
                    except:
                        pass
                        
                    processed_count += 1
                        
                    # 3. Process & Chunk
                    processed = self.processor.process_issue(issue_data)
                    
                    # Continuous Checkpointing
                    if processed_count > 0 and processed_count % 50 == 0:
                        self.state_manager.update_last_sync_time(new_sync_time)
                        logger.info(f"Checkpoint saved... Fetched {processed_count} issues so far.")
                        
                    if not processed:
                        continue
                        
                    parent_doc = processed["parent"]
                    chunks = processed["chunks"]
                    
                    logger.info(f"Indexing issue {issue_data['key']}: {len(chunks)} chunks")
                    
                    if not chunks:
                        continue

                    # 4. Generate Embeddings (Batching per issue for simplicity)
                    texts = [chunk["text"] for chunk in chunks]
                    embeddings = await ai_client.get_embeddings(texts)
                    
                    # 4.1 Generate Sparse Embeddings
                    sparse_indices, sparse_values = sparse_embedder.generate_sparse_vectors(texts)
                    
                    # 5. Prepare Payload & Upsert
                    vectors = []
                    payloads = []
                    
                    for i, chunk in enumerate(chunks):
                        vectors.append(embeddings[i])
                        # Flatten metadata for Qdrant payload
                        payload = chunk["metadata"]
                        payload["content"] = chunk["text"]
                        payload["doc_id"] = chunk["id"]
                        payloads.append(payload)
                    
                    self.vector_repo.upsert_parent_child(
                        parent_doc=parent_doc, 
                        child_documents=payloads, 
                        child_vectors=vectors,
                        child_sparse_indices=sparse_indices,
                        child_sparse_values=sparse_values
                    )
                        
                except Exception as e:
                    logger.error(f"Error processing issue {issue_data.get('key')}: {e}")
                    # Continue to next issue, but new_sync_time will only advance up to successful issues
            
            # 6. Update State if successful (or mostly successful)
            # using new_sync_time which tracked the Max time of successfully pushed issues, 
            # or current time if there were no issues fetched.
            self.state_manager.update_last_sync_time(new_sync_time)
            logger.info(f"Sync Cycle Completed. Processed {processed_count} issues.")
            
        except Exception as e:
            logger.error(f"Critical error in sync cycle: {e}")

    async def run_deletion_scrub(self):
        """Find and remove deleted items from Qdrant."""
        logger.info("Starting Jira Deletion Scrub Cycle...")
        try:
            source_keys = self.jira_client.fetch_all_active_keys()
            qdrant_keys = self.vector_repo.get_all_keys(source="jira")
            
            keys_to_delete = list(qdrant_keys - source_keys)
            if keys_to_delete:
                logger.info(f"Found {len(keys_to_delete)} deleted or ghost issues. Removing from Vector DB...")
                self.vector_repo.delete_by_keys(keys_to_delete)
                logger.info("Deletion scrub completed.")
            else:
                logger.info("No ghost issues found in Vector DB.")
        except Exception as e:
            logger.error(f"Critical error in deletion scrub cycle: {e}")

def job():
    """Wrapper to run async job synchronously for schedule."""
    bot = JiraBot()
    asyncio.run(bot.run_sync_cycle())

def scrub_job():
    bot = JiraBot()
    asyncio.run(bot.run_deletion_scrub())

if __name__ == "__main__":
    logger.info("Jira Bot Started")
    
    # Run once immediately on startup
    job()
    scrub_job()
    
    # Schedule every 10 minutes
    schedule.every(10).minutes.do(job)
    
    # Schedule scrub every 1 hour
    schedule.every(1).hours.do(scrub_job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
