# identification_service.py - Optimized identification service

import cv2
import logging
import numpy as np
from typing import Optional, Dict, Any, List
import time
import base64
import asyncio
from datetime import datetime

from detection.app.service.identification.identification_service import IdentificationDetectionSystem
from detection.app.service.video_streaming_client import VideoStreamingClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PieceIdentificationProcessor:
    """
    Optimized piece identification processor - completely separate from detection
    Focuses on identifying what pieces are present without lot validation
    """
    
    def __init__(self, confidence_threshold=0.5):
        self.detection_system = None
        self.is_initialized = False
        self.identification_confidence_threshold = confidence_threshold
        
        # Identification-specific statistics
        self.stats = {
            'identifications_performed': 0,
            'total_pieces_identified': 0,
            'unique_pieces_count': 0,
            'unique_pieces_list': [],
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'confidence_threshold': confidence_threshold,
            'device': 'unknown',
            'recent_identifications': []  # Last 10 identifications
        }
        
        self.video_client = VideoStreamingClient()
        
        # Cache for performance
        self._label_cache = {}
        self._frame_cache = {}
    
    async def initialize(self):
        """Initialize identification system"""
        try:
            if not self.is_initialized:
                logger.info("🚀 Initializing piece identification system...")
                
                loop = asyncio.get_event_loop()
                self.detection_system = await loop.run_in_executor(
                    None, 
                    lambda: IdentificationDetectionSystem(self.identification_confidence_threshold)
                )
                
                self.is_initialized = True
                self.stats['device'] = str(self.detection_system.device)
                
                logger.info(f"✅ Piece identification system initialized on device: {self.detection_system.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize identification system: {e}")
            raise
    
    async def identify_with_frame_capture(self, camera_id: int, freeze_stream: bool = True, quality: int = 85) -> Dict[str, Any]:
        """
        Main identification method - capture frame and identify all pieces
        """
        start_time = time.time()
        stream_frozen = False
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting identification for camera {camera_id}")
            
            # Freeze stream if requested
            if freeze_stream:
                freeze_success = await self.video_client.freeze_stream(camera_id)
                stream_frozen = freeze_success
            
            # Get current frame
            frame = await self.video_client.get_current_frame(camera_id)
            if frame is None:
                raise Exception(f"Could not get frame from camera {camera_id}")
            
            # Optimize frame
            frame = self._optimize_frame(frame)
            
            # Perform identification
            identification_result = await self._identify_pieces_async(frame)
            
            # Create annotated frame
            annotated_frame = await self._create_annotated_frame_async(frame, identification_result['pieces'])
            
            # Update frozen frame if needed
            if stream_frozen and annotated_frame is not None:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
                    if success:
                        await self.video_client.update_frozen_frame(camera_id, buffer.tobytes())
                except Exception as e:
                    logger.error(f"❌ Error updating frozen frame: {e}")
            
            # Encode frame for response
            frame_b64 = ""
            if annotated_frame is not None:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
                    if success:
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                except Exception as e:
                    logger.error(f"❌ Error encoding frame: {e}")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_stats(identification_result, processing_time)
            
            return {
                'success': True,
                'identification_result': identification_result,
                'frame_with_overlay': frame_b64,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': time.time(),
                'stream_frozen': stream_frozen
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error in identification: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'identification_result': {'pieces': [], 'total_pieces_found': 0, 'analysis_timestamp': time.time()},
                'frame_with_overlay': "",
                'processing_time_ms': round(processing_time, 2),
                'timestamp': time.time(),
                'stream_frozen': stream_frozen
            }
    
    async def _identify_pieces_async(self, frame: np.ndarray) -> Dict[str, Any]:
        """Async wrapper for piece identification"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.identify_pieces_in_frame, frame)
    
    def identify_pieces_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Identify all pieces in a frame - core identification logic
        """
        try:
            # Use the detailed identification method
            pieces = self.detection_system.get_detailed_identification_results(frame)
            
            return {
                'pieces': pieces,
                'total_pieces_found': len(pieces),
                'analysis_timestamp': time.time(),
                'confidence_threshold_used': self.identification_confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"❌ Error identifying pieces in frame: {e}")
            return {
                'pieces': [],
                'total_pieces_found': 0,
                'analysis_timestamp': time.time(),
                'error': str(e)
            }
    
    async def _create_annotated_frame_async(self, frame: np.ndarray, pieces: List[Dict]) -> Optional[np.ndarray]:
        """Async wrapper for frame annotation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._create_annotated_frame, frame, pieces)
    
    def _create_annotated_frame(self, frame: np.ndarray, pieces: List[Dict]) -> Optional[np.ndarray]:
        """Create annotated frame with identification overlays"""
        try:
            annotated_frame = frame.copy()
            
            for piece in pieces:
                bbox = piece['bounding_box']
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Draw bounding box (yellow for identification)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                # Create label with confidence
                label = f"{piece['label']}: {piece['confidence_percentage']}%"
                
                # Enhanced label positioning
                label_x, label_y = self._calculate_label_position(
                    x1, y1, x2, y2, label, annotated_frame.shape
                )
                
                # Draw label background
                self._draw_label_background(annotated_frame, label_x, label_y, label)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (label_x, label_y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"❌ Error creating annotated frame: {e}")
            return frame
    
    def _calculate_label_position(self, x1: int, y1: int, x2: int, y2: int, 
                                label: str, frame_shape: tuple) -> tuple:
        """Calculate optimal label position"""
        frame_height, frame_width = frame_shape[:2]
        
        # Get label dimensions
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        label_width, label_height = label_size
        
        # Default position: above the box
        label_x = x1
        label_y = y1 - 10
        
        # Boundary checks
        if label_y - label_height < 5:  # Not enough space above
            if y2 + label_height + 15 < frame_height:  # Space below
                label_y = y2 + label_height + 10
            else:  # Place inside box
                label_y = y1 + label_height + 5
        
        # Horizontal boundary check
        if label_x + label_width > frame_width - 5:
            label_x = max(5, frame_width - label_width - 5)
        elif label_x < 5:
            label_x = 5
        
        # Ensure within bounds
        label_y = max(label_height + 5, min(label_y, frame_height - 5))
        
        return label_x, label_y
    
    def _draw_label_background(self, frame: np.ndarray, x: int, y: int, label: str):
        """Draw semi-transparent background for label"""
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        label_width, label_height = label_size
        
        padding = 4
        bg_x1 = max(0, x - padding)
        bg_y1 = max(0, y - label_height - padding)
        bg_x2 = min(frame.shape[1], x + label_width + padding)
        bg_y2 = min(frame.shape[0], y + padding)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    def _optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame for identification"""
        # Resize if needed
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        
        # Ensure contiguous array
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        return frame
    
    def _update_stats(self, identification_result: Dict[str, Any], processing_time: float):
        """Update identification statistics"""
        pieces_found = identification_result['total_pieces_found']
        
        self.stats['identifications_performed'] += 1
        self.stats['total_pieces_identified'] += pieces_found
        self.stats['total_processing_time'] += processing_time
        self.stats['avg_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['identifications_performed']
        )
        
        # Update unique pieces
        for piece in identification_result['pieces']:
            label = piece['label']
            if label not in self.stats['unique_pieces_list']:
                self.stats['unique_pieces_list'].append(label)
                self.stats['unique_pieces_count'] = len(self.stats['unique_pieces_list'])
        
        # Update recent identifications (keep last 10)
        recent_record = {
            'timestamp': identification_result['analysis_timestamp'],
            'pieces_found': pieces_found,
            'labels': [p['label'] for p in identification_result['pieces']],
            'processing_time_ms': round(processing_time, 2)
        }
        
        self.stats['recent_identifications'].insert(0, recent_record)
        if len(self.stats['recent_identifications']) > 10:
            self.stats['recent_identifications'] = self.stats['recent_identifications'][:10]
    
    def get_identification_summary(self, pieces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate identification summary"""
        if not pieces:
            return {
                'total_pieces': 0,
                'unique_labels': [],
                'label_counts': {},
                'confidence_stats': {},
                'highest_confidence_piece': None
            }
        
        # Count labels
        label_counts = {}
        confidences = []
        
        for piece in pieces:
            label = piece['label']
            confidence = piece['confidence']
            
            label_counts[label] = label_counts.get(label, 0) + 1
            confidences.append(confidence)
        
        # Find highest confidence piece
        highest_confidence_piece = max(pieces, key=lambda p: p['confidence']) if pieces else None
        
        # Calculate confidence statistics
        confidence_stats = {}
        if confidences:
            confidence_stats = {
                'average': round(sum(confidences) / len(confidences), 3),
                'minimum': round(min(confidences), 3),
                'maximum': round(max(confidences), 3),
                'count_high_confidence': sum(1 for c in confidences if c >= 0.8),
                'count_medium_confidence': sum(1 for c in confidences if 0.5 <= c < 0.8),
                'count_low_confidence': sum(1 for c in confidences if c < 0.5)
            }
        
        return {
            'total_pieces': len(pieces),
            'unique_labels': list(label_counts.keys()),
            'label_counts': label_counts,
            'confidence_stats': confidence_stats,
            'highest_confidence_piece': {
                'label': highest_confidence_piece['label'],
                'confidence': highest_confidence_piece['confidence'],
                'confidence_percentage': highest_confidence_piece['confidence_percentage']
            } if highest_confidence_piece else None
        }
    
    def set_confidence_threshold(self, threshold: float) -> bool:
        """Update confidence threshold"""
        if 0.1 <= threshold <= 1.0:
            self.identification_confidence_threshold = threshold
            self.stats['confidence_threshold'] = threshold
            
            if self.detection_system:
                self.detection_system.set_confidence_threshold(threshold)
            
            logger.info(f"🎯 Confidence threshold updated to {threshold}")
            return True
        return False
    
    async def unfreeze_stream(self, camera_id: int) -> bool:
        """Unfreeze video stream"""
        return await self.video_client.unfreeze_stream(camera_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get identification statistics"""
        return {
            'is_initialized': self.is_initialized,
            **self.stats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.video_client.close()

# Global identification processor instance
piece_identification_processor = PieceIdentificationProcessor()  