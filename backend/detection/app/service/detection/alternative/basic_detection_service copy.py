# improved_basic_detection_service.py - Enhanced lot validation logic with lot context support

import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any, List
import time
import base64
import asyncio
from sqlalchemy.orm import Session, selectinload
from datetime import datetime
import json

# Import your detection system and database components
from detection.app.service.detection.detection_service import DetectionSystem
from detection.app.db.models.detectionLot import DetectionLot
from detection.app.db.models.detectionSession import DetectionSession
from detection.app.schema.lotResponse import LotResponse, DetectionResponse, LotValidationResult
from detection.app.schema.lotRequest import DetectionRequest, LotCreationRequest
from detection.app.service.video_streaming_client import VideoStreamingClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class BasicDetectionProcessor:
    """Detection processor with enhanced lot validation logic and lot context support"""
    
    def __init__(self):
        self.detection_system = None
        self.is_initialized = False
        
        # NEW: Lot context tracking
        self.current_lot_id = None
        self.current_piece_label = None
        self.current_lot_info = None
        self.is_initialized_for_lot = False
        self.lot_model_loaded = False
        
        self.stats = {
            'detections_performed': 0,
            'targets_detected': 0,
            'lot_matches': 0,
            'lot_mismatches': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'lots_created': 0,
            'lots_completed': 0
        }
        self.video_client = VideoStreamingClient()
    
    @property 
    def current_lot_context(self):
        """Get current lot context information"""
        return {
            'lot_id': self.current_lot_id,
            'piece_label': self.current_piece_label,
            'lot_name': self.current_lot_info.lot_name if self.current_lot_info else None,
            'expected_pieces': self.current_lot_info.expected_piece_number if self.current_lot_info else None,
            'is_initialized_for_lot': self.is_initialized_for_lot,
            'lot_model_loaded': self.lot_model_loaded
        }
    
    async def initialize(self, target_piece_label: Optional[str] = None):
        """
        Initialize basic detection system with optional target piece label
        
        Args:
            target_piece_label: Optional piece label to initialize the model for
        """
        try:
            if not self.is_initialized:
                logger.info("🚀 Initializing basic detection system...")
                
                if target_piece_label:
                    logger.info(f"🎯 Initializing with target piece: {target_piece_label}")
                else:
                    logger.info("⚠️ Initializing without specific target piece (generic model)")
                    # For backward compatibility, use a default piece label if none provided
                    # You might want to set this to a common piece label in your system
                    target_piece_label = "default"  # or None if your system supports generic models
                
                loop = asyncio.get_event_loop()
                self.detection_system = await loop.run_in_executor(
                    None, DetectionSystem, 0.5, target_piece_label  # confidence_threshold, target_piece_label
                )
                
                self.is_initialized = True
                logger.info(f"✅ Basic detection system initialized on device: {self.detection_system.device}")
            else:
                logger.info("ℹ️ Basic detection system already initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize basic detection system: {e}")
            self.clear_lot_context()  # Clear any partial context on failure
            raise

    async def initialize_with_lot_context(self, lot_id: int, piece_label: str, db: Session):
        """Initialize detection system with specific lot context"""
        try:
            logger.info(f"🎯 Initializing detection system for lot {lot_id} with piece: {piece_label}")
            
            # Get lot information from database
            lot_info = self.get_detection_lot(lot_id, db)
            if not lot_info:
                raise ValueError(f"Lot {lot_id} not found in database")
            
            # Initialize basic system first WITH the piece label
            await self.initialize(target_piece_label=piece_label)
            
            # Check if we need to switch contexts
            if (self.current_lot_id != lot_id or 
                self.current_piece_label != piece_label or 
                not self.is_initialized_for_lot):
                
                logger.info(f"🔄 Switching lot context from {self.current_lot_id}({self.current_piece_label}) to {lot_id}({piece_label})")
                
                # Update lot context
                self.current_lot_id = lot_id
                self.current_piece_label = piece_label  
                self.current_lot_info = lot_info
                
                # Switch model to the specific piece if detection system supports it
                if hasattr(self.detection_system, 'switch_model_for_piece'):
                    try:
                        model_switch_success = self.detection_system.switch_model_for_piece(piece_label)
                        if model_switch_success:
                            self.lot_model_loaded = True
                            logger.info(f"✅ Model successfully switched to piece: {piece_label}")
                        else:
                            logger.warning(f"⚠️ Model switch failed for piece: {piece_label}")
                            self.lot_model_loaded = False
                    except Exception as model_error:
                        logger.warning(f"⚠️ Error switching model for piece {piece_label}: {model_error}")
                        self.lot_model_loaded = False
                else:
                    # Detection system doesn't support model switching, use current model
                    logger.info("ℹ️ Detection system doesn't support model switching, using current model")
                    self.lot_model_loaded = True
                
                self.is_initialized_for_lot = True
                
            logger.info(f"✅ Detection system initialized for lot {lot_id} - piece: {piece_label}")
            
            return {
                'success': True,
                'message': f'Detection system ready for lot {lot_id} with piece: {piece_label}',
                'lot_context': self.current_lot_context
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize with lot context: {e}")
            # Clear context on failure
            self.clear_lot_context()
            raise
    
    def clear_lot_context(self):
        """Clear current lot context"""
        logger.info("🧹 Clearing lot context")
        self.current_lot_id = None
        self.current_piece_label = None
        self.current_lot_info = None
        self.is_initialized_for_lot = False
        self.lot_model_loaded = False
    
    def is_initialized_for_current_lot(self, lot_id: int, piece_label: str) -> bool:
        """Check if system is initialized for the specified lot and piece"""
        return (self.is_initialized_for_lot and
                self.current_lot_id == lot_id and
                self.current_piece_label == piece_label and
                self.lot_model_loaded)
    
    def validate_lot_against_detection(self, lot_info: LotResponse, detection_results: Dict[str, Any]) -> LotValidationResult:
        """
        Enhanced validation logic to check if detection results match lot expectations
        """
        errors = []
        
        # Extract detection data
        detected_target = detection_results.get('detected_target', False)
        correct_pieces_count = detection_results.get('correct_pieces_count', 0)
        non_target_count = detection_results.get('non_target_count', 0)
        total_pieces_detected = detection_results.get('total_pieces_detected', 0)
        confidence = detection_results.get('confidence', 0.0) or 0.0
        
        # Get expected values from lot
        expected_count = lot_info.expected_piece_number
        expected_label = self.current_piece_label or "target"  # Use current piece context
        
        # Validation logic
        is_valid = True
        
        # 1. Check if target label was detected
        if not detected_target:
            errors.append(f"Expected target label '{expected_label}' not detected")
            is_valid = False
        
        # 2. Check piece count matches exactly
        if correct_pieces_count != expected_count:
            errors.append(f"Piece count mismatch: expected {expected_count}, found {correct_pieces_count}")
            is_valid = False
        
        # 3. Check no incorrect/unexpected labels are present
        if non_target_count > 0:
            errors.append(f"Found {non_target_count} incorrect/unexpected pieces")
            is_valid = False
        
        # 4. Additional validation: total detected should match expected (no extra pieces)
        expected_total = expected_count  # Only correct pieces should be present
        if total_pieces_detected != expected_total:
            errors.append(f"Total pieces mismatch: expected {expected_total}, detected {total_pieces_detected}")
            is_valid = False
        
        # 5. Confidence threshold check (optional)
        min_confidence = 0.2  # Configurable threshold
        if confidence > 0 and confidence < min_confidence:
            errors.append(f"Low confidence score: {confidence:.2f} below threshold {min_confidence}")
            is_valid = False
        
        # Create detected labels list for reporting
        detected_labels = []
        if detected_target and correct_pieces_count > 0:
            detected_labels.extend([expected_label] * correct_pieces_count)
        if non_target_count > 0:
            detected_labels.extend(["incorrect_label"] * non_target_count)
        
        validation_result = LotValidationResult(
            is_valid=is_valid,
            expected_count=expected_count,
            actual_correct_count=correct_pieces_count,
            actual_incorrect_count=non_target_count,
            expected_label=expected_label,
            detected_labels=detected_labels,
            errors=errors,
            confidence_score=confidence
        )
        
        # Log validation result with lot context
        if is_valid:
            logger.info(f"✅ Lot {lot_info.lot_id} validation PASSED: {correct_pieces_count}/{expected_count} correct pieces for {expected_label}")
        else:
            logger.warning(f"❌ Lot {lot_info.lot_id} validation FAILED for {expected_label}: {', '.join(errors)}")
        
        return validation_result
    
    def create_detection_lot(self, lot_request: LotCreationRequest, db: Session) -> LotResponse:
        """Create a new detection lot with improved date handling"""
        try:
            # Create new detection lot - created_at will be set automatically by the database
            new_lot = DetectionLot(
                lot_name=lot_request.lot_name,
                expected_piece_id=lot_request.expected_piece_id,
                expected_piece_number=lot_request.expected_piece_number,
                is_target_match=False
            )
            
            db.add(new_lot)
            db.commit()
            db.refresh(new_lot)
            
            self.stats['lots_created'] += 1
            
            logger.info(f"📦 Created detection lot {new_lot.id}: '{lot_request.lot_name}' expecting {lot_request.expected_piece_number} pieces")
            
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
            lot.completed_at = datetime.utcnow() if is_match else None
            
            db.commit()
            
            if is_match:
                self.stats['lots_completed'] += 1
                self.stats['lot_matches'] += 1
                logger.info(f"✅ Lot {lot_id} marked as target match and completed!")
            else:
                self.stats['lot_mismatches'] += 1
                logger.info(f"🔄 Lot {lot_id} marked as not matching - needs correction")
            
            return True
                
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error updating lot target match: {e}")
            return False
    
    def create_detection_session(self, lot_id: int, detection_results: dict, validation_result: LotValidationResult, db: Session) -> int:
        """Create a detection session record with validation results"""
        try:
            # Calculate detection rate based on validation
            detection_rate = 1.0 if validation_result.is_valid else 0.0
            
            # Create detection session - created_at will be set automatically
            session = DetectionSession(
                lot_id=lot_id,
                correct_pieces_count=validation_result.actual_correct_count,
                misplaced_pieces_count=validation_result.actual_incorrect_count,
                total_pieces_detected=validation_result.actual_correct_count + validation_result.actual_incorrect_count,
                confidence_score=validation_result.confidence_score,
                is_target_match=validation_result.is_valid,
                detection_rate=detection_rate
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            logger.info(f"📊 Created detection session {session.id} for lot {lot_id} - Validation passed: {validation_result.is_valid}")
            
            return session.id
                
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error creating detection session: {e}")
            raise
    
    async def detect_with_lot_tracking(self, request: DetectionRequest, db: Session) -> DetectionResponse:
        """Perform detection with enhanced lot validation and session recording"""
        start_time = time.time()
        stream_frozen = False
        session_id = None
        lot_validation_result = None
        validation_errors = []
        
        try:
            # Ensure basic initialization
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting lot-tracked detection for camera {request.camera_id}, target: '{request.target_label}'")
            
            # Validate lot exists and initialize for lot context if lot_id provided
            lot_info = None
            if request.lot_id:
                lot_info = self.get_detection_lot(request.lot_id, db)
                if not lot_info:
                    raise Exception(f"Detection lot {request.lot_id} not found")
                
                # Initialize with lot context if not already initialized for this lot/piece
                if not self.is_initialized_for_current_lot(request.lot_id, request.target_label):
                    logger.info(f"🎯 Initializing for lot context: lot {request.lot_id}, piece {request.target_label}")
                    await self.initialize_with_lot_context(request.lot_id, request.target_label, db)
                
                logger.info(f"📦 Using lot {request.lot_id}: '{lot_info.lot_name}' expecting {lot_info.expected_piece_number} pieces of {request.target_label}")
            
            # Freeze the stream
            freeze_success = await self.video_client.freeze_stream(request.camera_id)
            if freeze_success:
                stream_frozen = True
            
            # Get current frame
            frame = await self.video_client.get_current_frame(request.camera_id)
            if frame is None:
                raise Exception(f"Could not get current frame from camera {request.camera_id}")
            
            # Ensure frame is contiguous
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Perform detection using the lot-aware detection system
            logger.info(f"🎯 Running detection on frame from camera {request.camera_id} - Original size: {frame.shape}")
            loop = asyncio.get_event_loop()
            detection_results = await loop.run_in_executor(
                None, 
                self.detection_system.detect_with_sliding_window, 
                frame, 
                request.target_label
            )
            
            # Parse detection results with better error handling
            if isinstance(detection_results, tuple) and len(detection_results) >= 6:
                processed_frame = detection_results[0]
                detected_target = detection_results[1]
                non_target_count = detection_results[2]
                total_pieces_detected = detection_results[3]
                correct_pieces_count = detection_results[4]
                confidence = detection_results[5]
                
                logger.info(f"🔍 Detection results - target: {detected_target}, "
                        f"correct: {correct_pieces_count}, "
                        f"incorrect: {non_target_count}, "
                        f"total: {total_pieces_detected}, "
                        f"confidence: {confidence}")
            else:
                logger.warning(f"❌ Unexpected detection results structure: {detection_results}")
                processed_frame = detection_results[0] if isinstance(detection_results, tuple) else detection_results
                detected_target = False
                non_target_count = 0
                total_pieces_detected = 0
                correct_pieces_count = 0
                confidence = 0

            # Prepare detection data for validation
            detection_session_data = {
                'detected_target': detected_target,
                'non_target_count': non_target_count,
                'total_pieces_detected': total_pieces_detected,
                'correct_pieces_count': correct_pieces_count,
                'confidence': confidence
            }
            
            # Perform lot validation if lot_id provided
            is_target_match = False
            if request.lot_id and lot_info:
                validation_result = self.validate_lot_against_detection(lot_info, detection_session_data)
                lot_validation_result = validation_result.to_dict()
                validation_errors = validation_result.errors
                is_target_match = validation_result.is_valid
                
                # Create detection session with validation results
                session_id = self.create_detection_session(request.lot_id, detection_session_data, validation_result, db)
                
                # Update lot target match status only if validation passes
                if validation_result.is_valid:
                    self.update_lot_target_match(request.lot_id, True, db)
                    logger.info(f"🎯 LOT VALIDATION PASSED: All criteria met for lot {request.lot_id}")
                else:
                    logger.warning(f"❌ LOT VALIDATION FAILED: {', '.join(validation_errors)}")
            else:
                # No lot validation, use simple detection result
                is_target_match = detected_target
            
            # Update frozen frame with results
            if processed_frame is not None and stream_frozen:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        await self.video_client.update_frozen_frame(request.camera_id, buffer.tobytes())
                except Exception as e:
                    logger.error(f"❌ Error updating frozen frame: {e}")
            
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
            
            # Calculate detection rate based on validation
            detection_rate = 1.0 if is_target_match else 0.0
            
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
                is_target_match=is_target_match,
                detection_rate=detection_rate,
                lot_validation_result=lot_validation_result,
                validation_errors=validation_errors
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
                detection_rate=0.0,
                lot_validation_result=None,
                validation_errors=["Detection processing failed"]
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
        """Get enhanced detection statistics including lot context"""
        base_stats = {
            'is_initialized': self.is_initialized,
            'device': str(self.detection_system.device) if self.detection_system else "unknown",
            **self.stats
        }
        
        # Add lot context information
        lot_context_stats = {
            'current_lot_context': self.current_lot_context,
            'is_initialized_for_lot': self.is_initialized_for_lot,
            'lot_model_loaded': self.lot_model_loaded
        }
        
        return {**base_stats, **lot_context_stats}
    
    async def cleanup(self):
        """Cleanup resources and clear lot context"""
        await self.video_client.close()
        self.clear_lot_context()
        
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
                
                lot_responses.append(LotResponse.from_db_model(lot, total_sessions, successful_detections))
            
            return lot_responses
                
        except Exception as e:
            logger.error(f"❌ Error getting all detection lots: {e}")
            return []

# Global enhanced detection processor
basic_detection_processor = BasicDetectionProcessor()