
import logging
import threading

from typing import Optional

from pydantic import BaseModel


import os

from concurrent.futures import ThreadPoolExecutor
import queue


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)




# Configuration
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"
SAVE_DIR = "captured_images"



# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=3)



if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)



class DetectionSession(BaseModel):
    session_id: str
    camera_id: int
    target_label: str
    detection_frequency: float = 5.0  # Hz
    confidence_threshold: float = 0.5
    save_detections: bool = False
    user_id: Optional[str] = None

class DetectionSessionManager:
    """Manages multiple detection sessions."""
    
    def __init__(self):
        self.sessions = {}
        self.stop_events = {}
        self.results_queues = {}
        
    def create_session(self, session: DetectionSession) -> str:
        """Create a new detection session."""
        self.sessions[session.session_id] = session
        self.stop_events[session.session_id] = threading.Event()
        self.results_queues[session.session_id] = queue.Queue(maxsize=10)
        return session.session_id
    
    def stop_session(self, session_id: str):
        """Stop a detection session."""
        if session_id in self.stop_events:
            self.stop_events[session_id].set()
            logger.info(f"Stop signal sent for detection session {session_id}")
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources."""
        for container in [self.sessions, self.stop_events, self.results_queues]:
            container.pop(session_id, None)
    
    def get_session(self, session_id: str) -> Optional[DetectionSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if session is active."""
        return session_id in self.sessions and not self.stop_events.get(session_id, threading.Event()).is_set()