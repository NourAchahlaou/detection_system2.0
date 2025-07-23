import logging
import asyncio
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class VideoStreamManager:
    """Manages video streaming with optimized performance."""
    
    def __init__(self):
        self.active_cameras: Dict[int, bool] = {}
        self.stop_events: Dict[int, asyncio.Event] = {}
        
    def is_camera_active(self, camera_id: int) -> bool:
        """Check if camera is currently streaming."""
        return camera_id in self.active_cameras and self.active_cameras[camera_id]
    
    async def stop_camera_stream(self, camera_id: int) -> bool:
        """Stop streaming for a specific camera."""
        try:
            if camera_id in self.stop_events:
                self.stop_events[camera_id].set()
                logger.info(f"Stop signal sent for camera {camera_id}")
            
            # Clean up tracking
            if camera_id in self.active_cameras:
                del self.active_cameras[camera_id]
            if camera_id in self.stop_events:
                del self.stop_events[camera_id]
                
            return True
        except Exception as e:
            logger.error(f"Error stopping camera stream {camera_id}: {e}")
            return False
    
    def add_camera_stream(self, camera_id: int, stop_event: asyncio.Event):
        """Add a camera to active streaming."""
        self.active_cameras[camera_id] = True
        self.stop_events[camera_id] = stop_event
        logger.info(f"Added camera {camera_id} to active streams")
    
    def get_stream_info(self, camera_id: int) -> Dict[str, Any]:
        """Get detailed information about a stream."""
        is_active = self.is_camera_active(camera_id)
        has_stop_event = camera_id in self.stop_events
        
        return {
            "camera_id": camera_id,
            "is_active": is_active,
            "has_stop_event": has_stop_event,
            "can_be_stopped": has_stop_event
        }
    
    def get_all_active_cameras(self) -> list:
        """Get list of all active camera IDs."""
        return [camera_id for camera_id, active in self.active_cameras.items() if active]
    
    async def cleanup_all_streams(self) -> int:
        """Clean up all active streams."""
        cleanup_count = 0
        
        # Stop all streams
        for camera_id in list(self.active_cameras.keys()):
            try:
                await self.stop_camera_stream(camera_id)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up camera {camera_id}: {e}")
        
        # Clear all tracking
        self.active_cameras.clear()
        self.stop_events.clear()
        
        logger.info(f"Cleaned up {cleanup_count} streams")
        return cleanup_count