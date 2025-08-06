# improved_basic_detection_service.py - Better date handling and JSON serialization

import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any, List
import time
import base64
import aiohttp
import asyncio
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, update, and_, func
from datetime import datetime
import json

# Import your detection system and database components
from detection.app.service.detection_service import DetectionSystem
from detection.app.db.models.detectionLot import DetectionLot
from detection.app.db.models.detectionSession import DetectionSession
from detection.app.db.session import get_session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
VIDEO_STREAMING_SERVICE_URL = "http://video_streaming:8000"

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class DetectionRequest:
    """Enhanced detection request structure"""
    camera_id: int
    target_label: str
    lot_id: Optional[int] = None
    expected_piece_id: Optional[int] = None
    expected_piece_number: Optional[int] = None
    timestamp: float = None
    quality: int = 85
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class DetectionResponse:
    """Enhanced detection response structure"""
    camera_id: int
    target_label: str
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    confidence: Optional[float]
    frame_with_overlay: str
    timestamp: float
    stream_frozen: bool
    lot_id: Optional[int] = None
    session_id: Optional[int] = None
    is_target_match: bool = False
    detection_rate: float = 0.0

@dataclass
class LotCreationRequest:
    """Request to create a new detection lot"""
    lot_name: str
    expected_piece_id: int
    expected_piece_number: int

@dataclass
class LotResponse:
    """Response after creating or updating a lot with proper date serialization"""
    lot_id: int
    lot_name: str
    expected_piece_id: int
    expected_piece_number: int
    is_target_match: bool
    created_at: str  # ISO format string instead of datetime
    completed_at: Optional[str] = None  # ISO format string instead of datetime
    total_sessions: int = 0
    successful_detections: int = 0

    @classmethod
    def from_db_model(cls, lot: DetectionLot, total_sessions: int = 0, successful_detections: int = 0):
        """Create LotResponse from database model with proper date conversion"""
        return cls(
            lot_id=lot.id,
            lot_name=lot.lot_name,
            expected_piece_id=lot.expected_piece_id,
            expected_piece_number=lot.expected_piece_number,
            is_target_match=lot.is_target_match,
            created_at=lot.created_at.isoformat() if lot.created_at else datetime.utcnow().isoformat(),
            completed_at=lot.completed_at.isoformat() if lot.completed_at else None,
            total_sessions=total_sessions,
            successful_detections=successful_detections
        )

class VideoStreamingClient:
    """HTTP client to communicate with video streaming service"""
    
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
        """Update frozen frame with detection results"""
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

class BasicDetectionProcessor:
    """Detection processor with improved database integration and date handling"""
    
    def __init__(self):
        self.detection_system = None
        self.is_initialized = False
        self.stats = {
            'detections_performed': 0,
            'targets_detected': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'lots_created': 0,
            'lots_completed': 0
        }
        self.video_client = VideoStreamingClient()
    
    async def initialize(self):
        """Initialize detection system"""
        try:
            if not self.is_initialized:
                logger.info("🚀 Initializing enhanced detection system...")
                
                loop = asyncio.get_event_loop()
                self.detection_system = await loop.run_in_executor(
                    None, DetectionSystem
                )
                
                self.is_initialized = True
                logger.info(f"✅ Enhanced detection system initialized on device: {self.detection_system.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced detection system: {e}")
            raise
    
    def create_detection_lot(self, lot_request: LotCreationRequest, db: Session) -> LotResponse:
        """Create a new detection lot with improved date handling"""
        try:
            # Create new detection lot - created_at will be set automatically by the database
            new_lot = DetectionLot(
                lot_name=lot_request.lot_name,
                expected_piece_id=lot_request.expected_piece_id,
                expected_piece_number=lot_request.expected_piece_number,
                is_target_match=False
                # Don't set created_at explicitly - let the database handle it with server_default=func.now()
            )
            
            db.add(new_lot)
            db.commit()
            db.refresh(new_lot)  # This will populate the created_at field from the database
            
            self.stats['lots_created'] += 1
            
            logger.info(f"📦 Created detection lot {new_lot.id}: '{lot_request.lot_name}' expecting piece {lot_request.expected_piece_number}")
            
            # Use the class method to properly convert dates to strings
            return LotResponse.from_db_model(new_lot, total_sessions=0, successful_detections=0)
                
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error creating detection lot: {e}")
            raise

    def get_detection_lot(self, lot_id: int, db: Session) -> Optional[LotResponse]:
        """Get detection lot by ID with session statistics and proper date handling"""
        try:
            # Get lot with sessions
            lot = db.query(DetectionLot).options(selectinload(DetectionLot.detection_sessions)).filter(DetectionLot.id == lot_id).first()
            
            if not lot:
                return None
            
            # Calculate statistics
            total_sessions = len(lot.detection_sessions)
            successful_detections = sum(1 for session in lot.detection_sessions if session.is_target_match)
            
            # Use the class method to properly convert dates to strings
            return LotResponse.from_db_model(lot, total_sessions, successful_detections)
                
        except Exception as e:
            logger.error(f"❌ Error getting detection lot {lot_id}: {e}")
            return None

    def update_lot_target_match(self, lot_id: int, is_match: bool, db: Session) -> bool:
        """Update lot target match status and completion time"""
        try:
            # Get the lot
            lot = db.query(DetectionLot).filter(DetectionLot.id == lot_id).first()
            if not lot:
                logger.error(f"Lot {lot_id} not found")
                return False
            
            # Update lot
            lot.is_target_match = is_match
            # Set completed_at to current UTC time if marking as match, otherwise None
            lot.completed_at = datetime.utcnow() if is_match else None
            
            db.commit()
            
            if is_match:
                self.stats['lots_completed'] += 1
                logger.info(f"✅ Lot {lot_id} marked as target match and completed!")
            else:
                logger.info(f"🔄 Lot {lot_id} marked as not matching - needs correction")
            
            return True
                
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error updating lot target match: {e}")
            return False
    
    def create_detection_session(self, lot_id: int, detection_results: dict, db: Session) -> int:
        """Create a detection session record with automatic timestamp"""
        try:
            # Calculate detection rate
            total_detected = detection_results.get('total_pieces_detected', 0)
            correct_count = detection_results.get('correct_pieces_count', 0)
            detection_rate = (correct_count / total_detected) if total_detected > 0 else 0.0
            
            # Determine if this detection matches the expected target
            is_target_match = detection_results.get('detected_target', False)
            
            # Create detection session - created_at will be set automatically
            session = DetectionSession(
                lot_id=lot_id,
                correct_pieces_count=correct_count,
                misplaced_pieces_count=detection_results.get('non_target_count', 0),
                total_pieces_detected=total_detected,
                confidence_score=detection_results.get('confidence', 0.0) or 0.0,
                is_target_match=is_target_match,
                detection_rate=detection_rate
                # Don't set created_at explicitly - let the database handle it
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            logger.info(f"📊 Created detection session {session.id} for lot {lot_id} - Target match: {is_target_match}")
            
            return session.id
                
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error creating detection session: {e}")
            raise
    
    async def detect_with_lot_tracking(self, request: DetectionRequest, db: Session) -> DetectionResponse:
        """Perform detection with lot tracking and session recording"""
        start_time = time.time()
        stream_frozen = False
        session_id = None
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting lot-tracked detection for camera {request.camera_id}, target: '{request.target_label}'")
            
            # Validate lot exists if lot_id provided
            lot_info = None
            if request.lot_id:
                lot_info = self.get_detection_lot(request.lot_id, db)
                if not lot_info:
                    raise Exception(f"Detection lot {request.lot_id} not found")
                logger.info(f"📦 Using lot {request.lot_id}: '{lot_info.lot_name}' expecting piece {lot_info.expected_piece_number}")
            
            # Freeze the stream
            freeze_success = await self.video_client.freeze_stream(request.camera_id)
            if freeze_success:
                stream_frozen = True
            
            # Get current frame
            frame = await self.video_client.get_current_frame(request.camera_id)
            if frame is None:
                raise Exception(f"Could not get current frame from camera {request.camera_id}")
            
            # Prepare frame
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Perform detection
            logger.info(f"🎯 Running detection on frame from camera {request.camera_id}")
            loop = asyncio.get_event_loop()
            detection_results = await loop.run_in_executor(
                None, 
                self.detection_system.detect_and_contour, 
                frame, 
                request.target_label
            )
            
            # Parse detection results
            processed_frame = None
            detected_target = False
            non_target_count = 0
            confidence = None
            
            if isinstance(detection_results, tuple):
                processed_frame = detection_results[0]
                detected_target = detection_results[1] if len(detection_results) > 1 else False
                non_target_count = detection_results[2] if len(detection_results) > 2 else 0
                confidence = detection_results[3] if len(detection_results) > 3 else None
            else:
                processed_frame = detection_results
            
            # Update frozen frame with results
            if processed_frame is not None and stream_frozen:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        await self.video_client.update_frozen_frame(request.camera_id, buffer.tobytes())
                except Exception as e:
                    logger.error(f"❌ Error updating frozen frame: {e}")
            
            # Create detection session if lot_id provided
            if request.lot_id:
                detection_session_data = {
                    'detected_target': detected_target,
                    'non_target_count': non_target_count,
                    'total_pieces_detected': 1 if detected_target else 0,
                    'correct_pieces_count': 1 if detected_target else 0,
                    'confidence': confidence
                }
                
                session_id = self.create_detection_session(request.lot_id, detection_session_data, db)
                
                # Check if we should update lot target match status
                if detected_target and lot_info:
                    self.update_lot_target_match(request.lot_id, True, db)
            
            # Encode frame for response
            frame_b64 = ""
            if processed_frame is not None:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                except Exception as e:
                    logger.error(f"❌ Error encoding frame: {e}")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.stats['detections_performed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['detections_performed']
            
            if detected_target:
                self.stats['targets_detected'] += 1
                logger.info(f"🎯 TARGET DETECTED: '{request.target_label}' found in camera {request.camera_id}!")
            
            # Calculate detection rate
            detection_rate = 1.0 if detected_target else 0.0
            
            response = DetectionResponse(
                camera_id=request.camera_id,
                target_label=request.target_label,
                detected_target=detected_target,
                non_target_count=non_target_count,
                processing_time_ms=round(processing_time, 2),
                confidence=confidence,
                frame_with_overlay=frame_b64,
                timestamp=time.time(),
                stream_frozen=stream_frozen,
                lot_id=request.lot_id,
                session_id=session_id,
                is_target_match=detected_target,
                detection_rate=detection_rate
            )
            
            logger.info(f"✅ Lot-tracked detection completed for camera {request.camera_id} in {processing_time:.2f}ms")
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error in lot-tracked detection for camera {request.camera_id}: {e}")
            
            return DetectionResponse(
                camera_id=request.camera_id,
                target_label=request.target_label,
                detected_target=False,
                non_target_count=0,
                processing_time_ms=round(processing_time, 2),
                confidence=None,
                frame_with_overlay="",
                timestamp=time.time(),
                stream_frozen=stream_frozen,
                lot_id=request.lot_id,
                session_id=session_id,
                is_target_match=False,
                detection_rate=0.0
            )
    
    def get_lot_sessions(self, lot_id: int, db: Session) -> List[Dict[str, Any]]:
        """Get all detection sessions for a lot with proper date serialization"""
        try:
            sessions = db.query(DetectionSession).filter(DetectionSession.lot_id == lot_id).order_by(DetectionSession.created_at.desc()).all()
            
            return [
                {
                    'session_id': session.id,
                    'correct_pieces_count': session.correct_pieces_count,
                    'misplaced_pieces_count': session.misplaced_pieces_count,
                    'total_pieces_detected': session.total_pieces_detected,
                    'confidence_score': session.confidence_score,
                    'is_target_match': session.is_target_match,
                    'detection_rate': session.detection_rate,
                    'created_at': session.created_at.isoformat() if session.created_at else None
                }
                for session in sessions
            ]
                
        except Exception as e:
            logger.error(f"❌ Error getting lot sessions: {e}")
            return []
    
    async def unfreeze_stream(self, camera_id: int) -> bool:
        """Unfreeze the stream to resume live video"""
        return await self.video_client.unfreeze_stream(camera_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return {
            'is_initialized': self.is_initialized,
            'device': str(self.detection_system.device) if self.detection_system else "unknown",
            **self.stats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.video_client.close()
        
    def get_all_detection_lots(self, db: Session) -> List[LotResponse]:
        """Get all detection lots with their statistics and proper date handling"""
        try:
            lots = db.query(DetectionLot).order_by(DetectionLot.created_at.desc()).all()
            
            lot_responses = []
            for lot in lots:
                # Get session count for this lot
                total_sessions = db.query(DetectionSession).filter(DetectionSession.lot_id == lot.id).count()
                successful_detections = db.query(DetectionSession).filter(
                    DetectionSession.lot_id == lot.id,
                    DetectionSession.is_target_match == True
                ).count()
                
                # Use the class method to properly convert dates to strings
                lot_responses.append(LotResponse.from_db_model(lot, total_sessions, successful_detections))
            
            return lot_responses
                
        except Exception as e:
            logger.error(f"❌ Error getting all detection lots: {e}")
            return []

# Global enhanced detection processor
basic_detection_processor = BasicDetectionProcessor()