import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any

STATE_FILE = "ingestion_state.json"

class StateManager:
    """
    Manages the state of the ingestion bot (e.g., last sync time).
    Persists to a local JSON file.
    """
    
    def __init__(self, file_path: str = STATE_FILE):
        self.file_path = file_path

    def load_state(self) -> Dict[str, Any]:
        """Load state from file, returning defaults if not found."""
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_state(self, state: Dict[str, Any]):
        """Save state to file."""
        with open(self.file_path, "w") as f:
            json.dump(state, f, indent=2)

    def get_last_sync_time(self, default_days_ago: int = 365) -> datetime:
        """Get last sync time, or default to N days ago."""
        state = self.load_state()
        timestamp_str = state.get("last_sync_time")
        
        if timestamp_str:
            return datetime.fromisoformat(timestamp_str)
        
        return datetime.now() - timedelta(days=default_days_ago)

    def update_last_sync_time(self, timestamp: datetime):
        """Update last sync time to the given timestamp."""
        state = self.load_state()
        state["last_sync_time"] = timestamp.isoformat()
        self.save_state(state)
