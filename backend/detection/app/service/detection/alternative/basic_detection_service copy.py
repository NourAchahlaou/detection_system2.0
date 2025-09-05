# enhanced_basic_detection_service.py - Enhanced with detected pieces storage
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
from detection.app.db.models.detectedPiece import DetectedPiece
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

class PieceMatcher:
    """Helper class to match detected labels with actual pieces in the database"""
    
    @staticmethod
    def get_piece_by_label(detected_label: str, db: Session) -> Optional[int]:
        """
        Find piece ID by matching detected label with piece labels in the database
        
        Args:
            detected_label: The label detected by the model (e.g., "G123.12345.123.12")
            db: Database session
            
        Returns:
            piece_id if found, None otherwise
        """
        try:
            # Import here to avoid circular imports
            from detection.app.db.models.piece import Piece  # Adjust import path as needed
            
            # Direct match first
            piece = db.query(Piece).filter(Piece.piece_label == detected_label).first()
            if piece:
                logger.info(f"Found exact match for piece label '{detected_label}' -> piece_id: {piece.id}")
                return piece.id
            
            # Fuzzy matching if no exact match
            # You can implement more sophisticated matching logic here
            similar_pieces = db.query(Piece).filter(
                Piece.piece_label.ilike(f"%{detected_label}%")
            ).all()
            
            if similar_pieces:
                # For now, return the first similar match
                # You could implement more sophisticated similarity scoring
                best_match = similar_pieces[0]
                logger.info(f"Found similar match for '{detected_label}' -> '{best_match.piece_label}' (piece_id: {best_match.id})")
                return best_match.id
                
            logger.warning(f"No piece found for detected label: '{detected_label}'")
            return None
            
        except Exception as e:
            logger.error(f"Error matching piece for label '{detected_label}': {e}")
            return None
    
    @staticmethod
    def get_all_piece_labels(db: Session) -> Dict[str, int]:
        """
        Get all piece labels mapped to their IDs for faster lookup
        
        Returns:
            Dictionary mapping piece_label -> piece_id
        """
        try:
            from detection.app.db.models.piece import Piece
            
            pieces = db.query(Piece.id, Piece.piece_label).all()
            return {piece.piece_label: piece.id for piece in pieces}
            
        except Exception as e:
            logger.error(f"Error getting all piece labels: {e}")
            return {}

class BasicDetectionProcessor:
    """Detection processor with enhanced lot validation logic and detected pieces storage"""
    
    def __init__(self):
        self.detection_system = None
        self.is_initialized = False
        
        # Lot context tracking
        self.current_lot_id = None
        self.current_piece_label = None
        self.current_lot_info = None
        self.is_initialized_for_lot = False
        self.lot_model_loaded = False
        
        # Piece matching cache for performance
        self.piece_label_cache = {}
        self.cache_last_updated = None
        self.cache_ttl = 300  # Cache for 5 minutes
        
        self.stats = {
            'detections_performed': 0,
            'targets_detected': 0,
            'lot_matches': 0,
            'lot_mismatches': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'lots_created': 0,
            'lots_completed': 0,
            'detected_pieces_stored': 0,
            'pieces_matched_with_id': 0,
            'pieces_without_id': 0
        }
        self.video_client = VideoStreamingClient()
    
    def refresh_piece_cache(self, db: Session):
        """Refresh the piece label cache if needed"""
        current_time = time.time()
        
        if (self.cache_last_updated is None or 
            current_time - self.cache_last_updated > self.cache_ttl):
            
            logger.info("Refreshing piece label cache...")
            self.piece_label_cache = PieceMatcher.get_all_piece_labels(db)
            self.cache_last_updated = current_time
            logger.info(f"Cached {len(self.piece_label_cache)} piece labels")
    
    def get_piece_id_for_label(self, detected_label: str, db: Session) -> Optional[int]:
        """
        Get piece ID for a detected label using cache first, then database lookup
        """
        # Refresh cache if needed
        self.refresh_piece_cache(db)
        
        # Check cache first
        if detected_label in self.piece_label_cache:
            piece_id = self.piece_label_cache[detected_label]
            logger.debug(f"Found piece_id {piece_id} for label '{detected_label}' in cache")
            return piece_id
        
        # Fall back to database lookup for fuzzy matching
        piece_id = PieceMatcher.get_piece_by_label(detected_label, db)
        
        # Update cache with the result
        if piece_id:
            self.piece_label_cache[detected_label] = piece_id
        
        return piece_id
    
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
        """Initialize basic detection system with optional target piece label"""
        try:
            if not self.is_initialized:
                logger.info("🚀 Initializing basic detection system...")
                
                if target_piece_label:
                    logger.info(f"🎯 Initializing with target piece: {target_piece_label}")
                else:
                    logger.info("⚠️ Initializing without specific target piece (generic model)")
                    target_piece_label = "default"
                
                loop = asyncio.get_event_loop()
                self.detection_system = await loop.run_in_executor(
                    None, DetectionSystem, 0.5, target_piece_label
                )
                
                self.is_initialized = True
                logger.info(f"✅ Basic detection system initialized on device: {self.detection_system.device}")
            else:
                logger.info("ℹ️ Basic detection system already initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize basic detection system: {e}")
            self.clear_lot_context()
            raise

    async def initialize_with_lot_context(self, lot_id: int, piece_label: str, db: Session):
        """Initialize detection system with specific lot context"""
        try:
            logger.info(f"🎯 Initializing detection system for lot {lot_id} with piece: {piece_label}")
            
            lot_info = self.get_detection_lot(lot_id, db)
            if not lot_info:
                raise ValueError(f"Lot {lot_id} not found in database")
            
            await self.initialize(target_piece_label=piece_label)
            
            if (self.current_lot_id != lot_id or 
                self.current_piece_label != piece_label or 
                not self.is_initialized_for_lot):
                
                logger.info(f"🔄 Switching lot context from {self.current_lot_id}({self.current_piece_label}) to {lot_id}({piece_label})")
                
                self.current_lot_id = lot_id
                self.current_piece_label = piece_label  
                self.current_lot_info = lot_info
                
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
    
    def extract_detected_pieces_from_results(self, detection_results: Dict[str, Any], target_label: str, db: Session) -> List[Dict[str, Any]]:
        """
        Extract individual detected pieces data from detection results with piece ID matching
        """
        detected_pieces = []
        
        # Check if detection_results contains detailed detection data
        if 'individual_detections' in detection_results:
            # Use the individual detections from the enhanced DetectionSystem
            for detection in detection_results['individual_detections']:
                # Get piece_id for this detection
                detected_label = detection.get('detected_label', 'unknown')
                piece_id = self.get_piece_id_for_label(detected_label, db)
                
                piece_data = {
                    'piece_id': piece_id,
                    'detected_label': detected_label,
                    'confidence_score': detection.get('confidence_score', 0.0),
                    'bounding_box_x1': detection.get('bounding_box_x1', 0),
                    'bounding_box_y1': detection.get('bounding_box_y1', 0),
                    'bounding_box_x2': detection.get('bounding_box_x2', 0),
                    'bounding_box_y2': detection.get('bounding_box_y2', 0),
                    'is_correct_piece': detection.get('is_correct_piece', False)
                }
                detected_pieces.append(piece_data)
                
                # Update stats
                if piece_id:
                    self.stats['pieces_matched_with_id'] += 1
                else:
                    self.stats['pieces_without_id'] += 1
        else:
            # Fallback: Create synthetic pieces based on counts (less ideal)
            logger.warning("Using fallback method for detected pieces - individual detection data not available")
            
            correct_count = detection_results.get('correct_pieces_count', 0)
            non_target_count = detection_results.get('non_target_count', 0)
            
            # Try to get piece_id for the target label
            target_piece_id = self.get_piece_id_for_label(target_label, db) if target_label else None
            
            # Create entries for correct pieces
            for i in range(correct_count):
                piece_data = {
                    'piece_id': target_piece_id,
                    'detected_label': target_label,
                    'confidence_score': detection_results.get('confidence', 0.0),
                    'bounding_box_x1': 0,  # Placeholder - would need actual coordinates
                    'bounding_box_y1': 0,
                    'bounding_box_x2': 0,
                    'bounding_box_y2': 0,
                    'is_correct_piece': True
                }
                detected_pieces.append(piece_data)
                
                if target_piece_id:
                    self.stats['pieces_matched_with_id'] += 1
                else:
                    self.stats['pieces_without_id'] += 1
            
            # Create entries for incorrect pieces
            for i in range(non_target_count):
                piece_data = {
                    'piece_id': None,  # Unknown piece type
                    'detected_label': 'incorrect_piece',  # Placeholder label
                    'confidence_score': detection_results.get('confidence', 0.0),
                    'bounding_box_x1': 0,  # Placeholder
                    'bounding_box_y1': 0,
                    'bounding_box_x2': 0,
                    'bounding_box_y2': 0,
                    'is_correct_piece': False
                }
                detected_pieces.append(piece_data)
                self.stats['pieces_without_id'] += 1
        
        logger.info(f"Extracted {len(detected_pieces)} pieces: "
                   f"{sum(1 for p in detected_pieces if p['piece_id'])} with piece_id, "
                   f"{sum(1 for p in detected_pieces if not p['piece_id'])} without piece_id")
        
        return detected_pieces
    
    def store_detected_pieces(self, session_id: int, detected_pieces_data: List[Dict[str, Any]], db: Session) -> List[int]:
        """Store detected pieces in the database and return list of created piece IDs"""
        try:
            created_piece_ids = []
            
            for piece_data in detected_pieces_data:
                detected_piece = DetectedPiece(
                    session_id=session_id,
                    piece_id=piece_data.get('piece_id'),  # Now properly populated
                    detected_label=piece_data['detected_label'],
                    confidence_score=piece_data['confidence_score'],
                    bounding_box_x1=piece_data['bounding_box_x1'],
                    bounding_box_y1=piece_data['bounding_box_y1'],
                    bounding_box_x2=piece_data['bounding_box_x2'],
                    bounding_box_y2=piece_data['bounding_box_y2'],
                    is_correct_piece=piece_data['is_correct_piece']
                )
                
                db.add(detected_piece)
                db.flush()  # Flush to get the ID
                created_piece_ids.append(detected_piece.id)
            
            db.commit()
            
            self.stats['detected_pieces_stored'] += len(detected_pieces_data)
            pieces_with_id = sum(1 for p in detected_pieces_data if p.get('piece_id'))
            logger.info(f"📊 Stored {len(detected_pieces_data)} detected pieces for session {session_id} "
                       f"({pieces_with_id} with piece_id)")
            
            return created_piece_ids
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error storing detected pieces: {e}")
            raise

    def validate_lot_against_detection(self, lot_info: LotResponse, detection_results: Dict[str, Any]) -> LotValidationResult:
        """Enhanced validation logic to check if detection results match lot expectations"""
        errors = []
        
        detected_target = detection_results.get('detected_target', False)
        correct_pieces_count = detection_results.get('correct_pieces_count', 0)
        non_target_count = detection_results.get('non_target_count', 0)
        total_pieces_detected = detection_results.get('total_pieces_detected', 0)
        confidence = detection_results.get('confidence', 0.0) or 0.0
        
        expected_count = lot_info.expected_piece_number
        expected_label = self.current_piece_label or "target"
        
        is_valid = True
        
        if not detected_target:
            errors.append(f"Expected target label '{expected_label}' not detected")
            is_valid = False
        
        if correct_pieces_count != expected_count:
            errors.append(f"Piece count mismatch: expected {expected_count}, found {correct_pieces_count}")
            is_valid = False
        
        if non_target_count > 0:
            errors.append(f"Found {non_target_count} incorrect/unexpected pieces")
            is_valid = False
        
        expected_total = expected_count
        if total_pieces_detected != expected_total:
            errors.append(f"Total pieces mismatch: expected {expected_total}, detected {total_pieces_detected}")
            is_valid = False
        
        min_confidence = 0.2
        if confidence > 0 and confidence < min_confidence:
            errors.append(f"Low confidence score: {confidence:.2f} below threshold {min_confidence}")
            is_valid = False
        
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
        
        if is_valid:
            logger.info(f"✅ Lot {lot_info.lot_id} validation PASSED: {correct_pieces_count}/{expected_count} correct pieces for {expected_label}")
        else:
            logger.warning(f"❌ Lot {lot_info.lot_id} validation FAILED for {expected_label}: {', '.join(errors)}")
        
        return validation_result
    
    def create_detection_lot(self, lot_request: LotCreationRequest, db: Session) -> LotResponse:
        """Create a new detection lot with improved date handling"""
        try:
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
            lot = db.query(DetectionLot).options(selectinload(DetectionLot.detection_sessions)).filter(DetectionLot.id == lot_id).first()
            
            if not lot:
                return None
            
            total_sessions = len(lot.detection_sessions)
            successful_detections = sum(1 for session in lot.detection_sessions if session.is_target_match)
            
            return LotResponse.from_db_model(lot, total_sessions, successful_detections)
                
        except Exception as e:
            logger.error(f"❌ Error getting detection lot {lot_id}: {e}")
            return None

    def update_lot_target_match(self, lot_id: int, is_match: bool, db: Session) -> bool:
        """Update lot target match status and completion time"""
        try:
            lot = db.query(DetectionLot).filter(DetectionLot.id == lot_id).first()
            if not lot:
                logger.error(f"Lot {lot_id} not found")
                return False
            
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
            detection_rate = 1.0 if validation_result.is_valid else 0.0
            
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
            
            # Extract and store detected pieces WITH piece_id matching
            target_label = self.current_piece_label or "target"
            detected_pieces_data = self.extract_detected_pieces_from_results(detection_results, target_label, db)
            
            if detected_pieces_data:
                created_piece_ids = self.store_detected_pieces(session.id, detected_pieces_data, db)
                logger.info(f"📊 Created detection session {session.id} with {len(created_piece_ids)} detected pieces")
            else:
                logger.info(f"📊 Created detection session {session.id} with no detected pieces")
            
            return session.id
                
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error creating detection session: {e}")
            raise
    def get_lot_sessions(self, lot_id: int, db: Session) -> List[Dict[str, Any]]:
        """Get all detection sessions for a lot with detected pieces data"""
        try:
            sessions = db.query(DetectionSession).options(
                selectinload(DetectionSession.detected_pieces)
            ).filter(DetectionSession.lot_id == lot_id).order_by(DetectionSession.created_at.desc()).all()
            
            sessions_data = []
            for session in sessions:
                # Get detected pieces data
                detected_pieces = []
                for piece in session.detected_pieces:
                    piece_data = {
                        'id': piece.id,
                        'piece_id': piece.piece_id,
                        'detected_label': piece.detected_label,
                        'confidence_score': piece.confidence_score,
                        'bounding_box': {
                            'x1': piece.bounding_box_x1,
                            'y1': piece.bounding_box_y1,
                            'x2': piece.bounding_box_x2,
                            'y2': piece.bounding_box_y2
                        },
                        'is_correct_piece': piece.is_correct_piece,
                        'created_at': piece.created_at.isoformat() if piece.created_at else None
                    }
                    detected_pieces.append(piece_data)
                
                session_data = {
                    'session_id': session.id,
                    'correct_pieces_count': session.correct_pieces_count,
                    'misplaced_pieces_count': session.misplaced_pieces_count,
                    'total_pieces_detected': session.total_pieces_detected,
                    'confidence_score': session.confidence_score,
                    'is_target_match': session.is_target_match,
                    'detection_rate': session.detection_rate,
                    'created_at': session.created_at.isoformat() if session.created_at else None,
                    'detected_pieces': detected_pieces,
                    'detected_pieces_count': len(detected_pieces)
                }
                sessions_data.append(session_data)
            
            return sessions_data
                
        except Exception as e:
            logger.error(f"❌ Error getting lot sessions: {e}")
            return []

    # Rest of the methods remain the same...
    async def detect_with_lot_tracking(self, request: DetectionRequest, db: Session) -> DetectionResponse:
        """Perform detection with enhanced lot validation and session recording"""
        start_time = time.time()
        stream_frozen = False
        session_id = None
        lot_validation_result = None
        validation_errors = []
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting lot-tracked detection for camera {request.camera_id}, target: '{request.target_label}'")
            
            lot_info = None
            if request.lot_id:
                lot_info = self.get_detection_lot(request.lot_id, db)
                if not lot_info:
                    raise Exception(f"Detection lot {request.lot_id} not found")
                
                if not self.is_initialized_for_current_lot(request.lot_id, request.target_label):
                    logger.info(f"🎯 Initializing for lot context: lot {request.lot_id}, piece {request.target_label}")
                    await self.initialize_with_lot_context(request.lot_id, request.target_label, db)
                
                logger.info(f"📦 Using lot {request.lot_id}: '{lot_info.lot_name}' expecting {lot_info.expected_piece_number} pieces of {request.target_label}")
            
            freeze_success = await self.video_client.freeze_stream(request.camera_id)
            if freeze_success:
                stream_frozen = True
            
            frame = await self.video_client.get_current_frame(request.camera_id)
            if frame is None:
                raise Exception(f"Could not get current frame from camera {request.camera_id}")
            
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            logger.info(f"🎯 Running detection on frame from camera {request.camera_id} - Original size: {frame.shape}")
            loop = asyncio.get_event_loop()
            detection_results = await loop.run_in_executor(
                None, 
                self.detection_system.detect_with_sliding_window, 
                frame, 
                request.target_label
            )
            
            # Updated unpacking to handle the 7 values returned by visualize_detections
            if isinstance(detection_results, tuple) and len(detection_results) >= 7:
                processed_frame = detection_results[0]
                detected_target = detection_results[1]
                non_target_count = detection_results[2]
                total_pieces_detected = detection_results[3]
                correct_pieces_count = detection_results[4]
                confidence = detection_results[5]
                individual_detections = detection_results[6]  # Now we capture this!
                
                logger.info(f"🔍 Detection results - target: {detected_target}, "
                        f"correct: {correct_pieces_count}, "
                        f"incorrect: {non_target_count}, "
                        f"total: {total_pieces_detected}, "
                        f"confidence: {confidence}, "
                        f"individual_detections: {len(individual_detections)}")
            elif isinstance(detection_results, tuple) and len(detection_results) >= 6:
                # Fallback for older format without individual_detections
                processed_frame = detection_results[0]
                detected_target = detection_results[1]
                non_target_count = detection_results[2]
                total_pieces_detected = detection_results[3]
                correct_pieces_count = detection_results[4]
                confidence = detection_results[5]
                individual_detections = []  # Empty list as fallback
                
                logger.warning(f"⚠️ Using fallback - individual detections not available")
            else:
                logger.warning(f"❌ Unexpected detection results structure: {detection_results}")
                processed_frame = detection_results[0] if isinstance(detection_results, tuple) else detection_results
                detected_target = False
                non_target_count = 0
                total_pieces_detected = 0
                correct_pieces_count = 0
                confidence = 0
                individual_detections = []

            # Create detection session data with individual detections
            detection_session_data = {
                'detected_target': detected_target,
                'non_target_count': non_target_count,
                'total_pieces_detected': total_pieces_detected,
                'correct_pieces_count': correct_pieces_count,
                'confidence': confidence,
                'individual_detections': individual_detections  # Add this key data
            }
            
            is_target_match = False
            if request.lot_id and lot_info:
                validation_result = self.validate_lot_against_detection(lot_info, detection_session_data)
                lot_validation_result = validation_result.to_dict()
                validation_errors = validation_result.errors
                is_target_match = validation_result.is_valid
                
                session_id = self.create_detection_session(request.lot_id, detection_session_data, validation_result, db)
                
                if validation_result.is_valid:
                    self.update_lot_target_match(request.lot_id, True, db)
                    logger.info(f"🎯 LOT VALIDATION PASSED: All criteria met for lot {request.lot_id}")
                else:
                    logger.warning(f"❌ LOT VALIDATION FAILED: {', '.join(validation_errors)}")
            else:
                is_target_match = detected_target
            
            if processed_frame is not None and stream_frozen:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, request.quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        await self.video_client.update_frozen_frame(request.camera_id, buffer.tobytes())
                except Exception as e:
                    logger.error(f"❌ Error updating frozen frame: {e}")
            
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
            
            self.stats['detections_performed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['detections_performed']
            
            if detected_target:
                self.stats['targets_detected'] += 1
            
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