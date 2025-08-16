# identification_service.py - Piece identification service without lot tracking

import cv2
import logging
import numpy as np
from typing import Dict, Any, List
import time
import base64
import asyncio
from datetime import datetime

# Import the detection system (reusing the same model)
from detection.app.service.detection.detection_service import DetectionSystem
from detection.app.service.video_streaming_client import VideoStreamingClient
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration



class PieceIdentificationSystem:
    """System for identifying pieces without lot tracking"""
    
    def __init__(self):
        self.detection_system = None
        self.is_initialized = False
        self.stats = {
            'identifications_performed': 0,
            'pieces_identified': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'unique_pieces_identified': set(),
            'identification_history': []
        }
        self.video_client = VideoStreamingClient()
        
        # Confidence threshold for identification (can be different from detection)
        self.identification_confidence_threshold = 0.5
    
    async def initialize(self):
        """Initialize identification system"""
        try:
            if not self.is_initialized:
                logger.info("🚀 Initializing piece identification system...")
                
                loop = asyncio.get_event_loop()
                self.detection_system = await loop.run_in_executor(
                    None, DetectionSystem, self.identification_confidence_threshold
                )
                
                self.is_initialized = True
                logger.info(f"✅ Piece identification system initialized on device: {self.detection_system.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize identification system: {e}")
            raise
    
    def identify_pieces_in_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Identify all pieces in the frame without lot validation
        Returns detailed information about each piece found
        """
        try:
            # Use the same detection method but focus on identification
            results = self.detection_system.detect_and_contour(frame, "")  # Empty target since we want all pieces
            
            if isinstance(results, tuple) and len(results) >= 6:
                processed_frame = results[0]
                detected_target = results[1]
                non_target_count = results[2]
                total_pieces_detected = results[3]
                correct_pieces_count = results[4]
                confidence = results[5]
            else:
                processed_frame = results[0] if isinstance(results, tuple) else results
                detected_target = False
                non_target_count = 0
                total_pieces_detected = 0
                correct_pieces_count = 0
                confidence = 0
            
            # For identification, we need to get detailed piece information
            piece_details = self._extract_piece_details(frame)
            
            identification_result = {
                'total_pieces_found': len(piece_details),
                'pieces': piece_details,
                'processed_frame': processed_frame,
                'overall_confidence': confidence,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            # Update stats
            self.stats['pieces_identified'] += len(piece_details)
            for piece in piece_details:
                self.stats['unique_pieces_identified'].add(piece['label'])
            
            # Add to history (keep last 100 identifications)
            self.stats['identification_history'].append({
                'timestamp': identification_result['analysis_timestamp'],
                'pieces_count': len(piece_details),
                'pieces_found': [p['label'] for p in piece_details]
            })
            if len(self.stats['identification_history']) > 100:
                self.stats['identification_history'].pop(0)
            
            logger.info(f"🔍 Identified {len(piece_details)} pieces in frame")
            
            return identification_result
            
        except Exception as e:
            logger.error(f"❌ Error identifying pieces: {e}")
            return {
                'total_pieces_found': 0,
                'pieces': [],
                'processed_frame': frame,
                'overall_confidence': 0,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def _extract_piece_details(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract detailed information about each piece in the frame
        This method performs a fresh detection to get individual piece data
        """
        try:
            # Convert the frame to tensor and run detection
            import torch
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).float().to(self.detection_system.device)
            frame_tensor /= 255.0
            frame_tensor = frame_tensor.half() if self.detection_system.device.type == 'cuda' else frame_tensor
            
            results = self.detection_system.model(frame_tensor.unsqueeze(0))[0]
            
            pieces = []
            class_names = self.detection_system.model.names
            
            if results.boxes is None or len(results.boxes) == 0:
                return pieces
            
            for i, box in enumerate(results.boxes):
                confidence = box.conf.item()
                
                if confidence < self.identification_confidence_threshold:
                    continue
                
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                class_id = int(box.cls.item())
                piece_label = class_names[class_id]
                
                # Calculate piece area and center
                width = x2 - x1
                height = y2 - y1
                area = width * height
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                piece_info = {
                    'piece_id': f"piece_{i}",
                    'label': piece_label,
                    'confidence': round(confidence, 3),
                    'bounding_box': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': width, 'height': height
                    },
                    'area': area,
                    'center': {'x': center_x, 'y': center_y},
                    'confidence_level': self._get_confidence_level(confidence)
                }
                
                pieces.append(piece_info)
            
            # Sort pieces by confidence (highest first)
            pieces.sort(key=lambda x: x['confidence'], reverse=True)
            
            return pieces
            
        except Exception as e:
            logger.error(f"❌ Error extracting piece details: {e}")
            return []
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to human-readable level"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    async def identify_with_frame_capture(self, camera_id: int, freeze_stream: bool = True, quality: int = 85) -> Dict[str, Any]:
        """
        Capture frame and identify pieces
        
        Args:
            camera_id: Camera to capture from
            freeze_stream: Whether to freeze the stream during identification
            quality: JPEG quality for frame encoding
        """
        start_time = time.time()
        stream_frozen = False
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Starting piece identification for camera {camera_id}")
            
            # Freeze stream if requested
            if freeze_stream:
                freeze_success = await self.video_client.freeze_stream(camera_id)
                if freeze_success:
                    stream_frozen = True
            
            # Get current frame
            frame = await self.video_client.get_current_frame(camera_id)
            if frame is None:
                raise Exception(f"Could not get current frame from camera {camera_id}")
            
            # Prepare frame
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Perform identification
            identification_result = self.identify_pieces_in_frame(frame)
            
            # Update frozen frame with identification results
            if identification_result['processed_frame'] is not None and stream_frozen:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode('.jpg', identification_result['processed_frame'], encode_params)
                    if success:
                        await self.video_client.update_frozen_frame(camera_id, buffer.tobytes())
                except Exception as e:
                    logger.error(f"❌ Error updating frozen frame: {e}")
            
            # Encode frame for response
            frame_b64 = ""
            if identification_result['processed_frame'] is not None:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode('.jpg', identification_result['processed_frame'], encode_params)
                    if success:
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                except Exception as e:
                    logger.error(f"❌ Error encoding frame: {e}")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.stats['identifications_performed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['identifications_performed']
            
            # Prepare response
            response = {
                'camera_id': camera_id,
                'identification_result': identification_result,
                'processing_time_ms': round(processing_time, 2),
                'frame_with_overlay': frame_b64,
                'timestamp': time.time(),
                'stream_frozen': stream_frozen,
                'success': True
            }
            
            logger.info(f"✅ Piece identification completed for camera {camera_id} in {processing_time:.2f}ms - Found {identification_result['total_pieces_found']} pieces")
            
            return response
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Error in piece identification for camera {camera_id}: {e}")
            
            return {
                'camera_id': camera_id,
                'identification_result': {
                    'total_pieces_found': 0,
                    'pieces': [],
                    'processed_frame': None,
                    'overall_confidence': 0,
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                },
                'processing_time_ms': round(processing_time, 2),
                'frame_with_overlay': "",
                'timestamp': time.time(),
                'stream_frozen': stream_frozen,
                'success': False,
                'error': str(e)
            }
    
    def get_identification_summary(self, pieces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of identified pieces"""
        if not pieces:
            return {
                'total_pieces': 0,
                'unique_labels': [],
                'label_counts': {},
                'average_confidence': 0,
                'highest_confidence_piece': None
            }
        
        label_counts = {}
        total_confidence = 0
        highest_confidence_piece = pieces[0]  # pieces are sorted by confidence
        
        for piece in pieces:
            label = piece['label']
            label_counts[label] = label_counts.get(label, 0) + 1
            total_confidence += piece['confidence']
        
        return {
            'total_pieces': len(pieces),
            'unique_labels': list(label_counts.keys()),
            'label_counts': label_counts,
            'average_confidence': round(total_confidence / len(pieces), 3),
            'highest_confidence_piece': highest_confidence_piece
        }
    
    async def unfreeze_stream(self, camera_id: int) -> bool:
        """Unfreeze the stream to resume live video"""
        return await self.video_client.unfreeze_stream(camera_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get identification statistics"""
        return {
            'is_initialized': self.is_initialized,
            'device': str(self.detection_system.device) if self.detection_system else "unknown",
            'confidence_threshold': self.identification_confidence_threshold,
            'identifications_performed': self.stats['identifications_performed'],
            'total_pieces_identified': self.stats['pieces_identified'],
            'unique_pieces_count': len(self.stats['unique_pieces_identified']),
            'unique_pieces_list': list(self.stats['unique_pieces_identified']),
            'avg_processing_time': self.stats['avg_processing_time'],
            'recent_identifications': self.stats['identification_history'][-10:]  # Last 10
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold for identification"""
        if 0.1 <= threshold <= 1.0:
            self.identification_confidence_threshold = threshold
            if self.detection_system:
                self.detection_system.confidence_threshold = threshold
            logger.info(f"🔧 Identification confidence threshold updated to {threshold}")
            return True
        else:
            logger.warning(f"❌ Invalid confidence threshold: {threshold}. Must be between 0.1 and 1.0")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.video_client.close()

# Global identification processor
piece_identification_processor = PieceIdentificationSystem()