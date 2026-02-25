import time
import schedule
import logging
from datetime import datetime
import asyncio
from typing import List, Dict, Any

from shared.config import settings
from shared.apis.openai_client import ai_client
from shared.apis.sparse_embedder import sparse_embedder
from shared.repositories.vector_store import ConfluenceRepository
from ingestion_bot.confluence_client import ConfluenceIngestionClient
from ingestion_bot.confluence_processor import ConfluenceProcessor
from ingestion_bot.state_manager import StateManager

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("confluence_ingestion.log")
    ]
)
logger = logging.getLogger("ConfluenceBot")

class ConfluenceBot:
    def __init__(self):
        self.confluence_client = ConfluenceIngestionClient()
        self.processor = ConfluenceProcessor()
        self.vector_repo = ConfluenceRepository()
        self.state_manager = StateManager() # Consider separating state file if needed
        # Overriding state manager internally for confluence so it doesn't clash with Jira
        self.state_manager.STATE_FILE = "confluence_sync_state.json"
        
    async def run_sync_cycle(self):
        """Execute one full sync cycle for Confluence."""
        logger.info("Starting Confluence Sync Cycle...")
        
        last_sync = self.state_manager.get_last_sync_time(default_days_ago=365)
        logger.info(f"Last sync time: {last_sync}")
        
        new_sync_time = last_sync
        processed_count = 0
        
        try:
            for page_data in self.confluence_client.fetch_updated_pages(last_sync):
                try:
                    # Track successful process time FIRST
                    from dateutil.parser import parse
                    try:
                        page_time = parse(page_data['updated']).replace(tzinfo=None)
                        if page_time > new_sync_time:
                            new_sync_time = page_time
                    except Exception:
                        pass
                        
                    processed_count += 1
                    
                    # Continuous Checkpointing for massive initial sync
                    if processed_count > 0 and processed_count % 50 == 0:
                        self.state_manager.update_last_sync_time(new_sync_time)
                        logger.info(f"Checkpoint saved... Fetched {processed_count} pages so far.")
                        
                    processed = self.processor.process_page(page_data)
                    parent_doc = processed["parent"]
                    chunks = processed["chunks"]
                    
                    logger.info(f"Indexing page {page_data['key']}: {len(chunks)} chunks")
                    if not chunks:
                        continue

                    # Generate Embeddings
                    texts = [chunk["text"] for chunk in chunks]
                    embeddings = await ai_client.get_embeddings(texts)
                    
                    # Generate Sparse Embeddings
                    sparse_indices, sparse_values = sparse_embedder.generate_sparse_vectors(texts)
                    
                    # Prepare Payload & Upsert
                    vectors = []
                    payloads = []
                    
                    for i, chunk in enumerate(chunks):
                        vectors.append(embeddings[i])
                        payload = chunk["metadata"]
                        payload["content"] = chunk["text"]
                        payload["doc_id"] = chunk["id"]
                        payloads.append(payload)
                    
                    await self.vector_repo.upsert_parent_child(
                        parent_doc=parent_doc, 
                        child_documents=payloads, 
                        child_vectors=vectors,
                        child_sparse_indices=sparse_indices,
                        child_sparse_values=sparse_values
                    )
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_data.get('key')}: {e}")
                    
            self.state_manager.update_last_sync_time(new_sync_time)
            logger.info(f"Confluence Sync Cycle Completed. Processed {processed_count} pages.")
            
        except Exception as e:
            logger.error(f"Critical error in Confluence sync cycle: {e}")

    async def run_deletion_scrub(self):
        """Find and remove deleted items from Qdrant."""
        logger.info("Starting Confluence Deletion Scrub Cycle...")
        try:
            source_keys = self.confluence_client.fetch_all_active_keys()
            qdrant_keys = await self.vector_repo.get_all_keys(source="confluence")
            
            keys_to_delete = list(qdrant_keys - source_keys)
            if keys_to_delete:
                logger.info(f"Found {len(keys_to_delete)} deleted or ghost pages. Removing from Vector DB...")
                await self.vector_repo.delete_by_keys(keys_to_delete)
                logger.info("Deletion scrub completed.")
            else:
                logger.info("No ghost pages found in Vector DB.")
        except Exception as e:
            logger.error(f"Critical error in deletion scrub cycle: {e}")

def job():
    bot = ConfluenceBot()
    asyncio.run(bot.run_sync_cycle())

def scrub_job():
    bot = ConfluenceBot()
    asyncio.run(bot.run_deletion_scrub())

if __name__ == "__main__":
    logger.info("Confluence Ingestion Bot Started")
    job()
    scrub_job()
    schedule.every(10).minutes.do(job)
    schedule.every(1).hours.do(scrub_job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
