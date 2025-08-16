# improved_basic_detection_service.py - Enhanced lot validation logic

import cv2
import logging
import numpy as np
from typing import Optional
import aiohttp


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
VIDEO_STREAMING_SERVICE_URL = "http://video_streaming:8000"

class VideoStreamingClient:
    """HTTP client to communicate with video streaming service for identification"""
    
    def __init__(self, base_url: str = VIDEO_STREAMING_SERVICE_URL):
        self.base_url = base_url
        self.session = None
    
    async def _get_session(self):
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def get_current_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Get current frame from video streaming service"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/current-frame"
            
            async with session.get(url) as response:
                if response.status == 200:
                    frame_data = await response.read()
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return frame
                else:
                    logger.error(f"Failed to get frame: HTTP {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting current frame: {e}")
            return None
    
    async def freeze_stream(self, camera_id: int) -> bool:
        """Freeze video stream"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/freeze"
            
            async with session.post(url) as response:
                if response.status == 200:
                    logger.info(f"🧊 Stream frozen for camera {camera_id}")
                    return True
                else:
                    logger.error(f"Failed to freeze stream: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error freezing stream: {e}")
            return False
    
    async def unfreeze_stream(self, camera_id: int) -> bool:
        """Unfreeze video stream"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/unfreeze"
            
            async with session.post(url) as response:
                if response.status == 200:
                    logger.info(f"🔥 Stream unfrozen for camera {camera_id}")
                    return True
                else:
                    logger.error(f"Failed to unfreeze stream: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error unfreezing stream: {e}")
            return False
    
    async def update_frozen_frame(self, camera_id: int, frame_bytes: bytes) -> bool:
        """Update frozen frame with identification results"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/video/basic/stream/{camera_id}/update-frozen-frame"
            
            data = aiohttp.FormData()
            data.add_field('frame', frame_bytes, content_type='image/jpeg')
            
            async with session.post(url, data=data) as response:
                if response.status == 200:
                    logger.info(f"✅ Updated frozen frame for camera {camera_id}")
                    return True
                else:
                    logger.error(f"Failed to update frozen frame: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating frozen frame: {e}")
            return False
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()