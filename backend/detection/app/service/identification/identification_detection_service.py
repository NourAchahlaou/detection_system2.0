# identification_service.py - Modified to allow initialization without group

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
    Complete piece identification processor following detection service pattern
    Focuses on identifying what pieces are present for a specific group
    Modified to allow initialization without group selection
    """
    
    def __init__(self, confidence_threshold=0.5):
        self.detection_system = None
        self.is_initialized = False  # General initialization
        self.is_group_loaded = False  # Group-specific model loaded
        self.identification_confidence_threshold = confidence_threshold
        self.current_group_name = None
        
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
            'current_group': None,
            'recent_identifications': []  # Last 10 identifications
        }
        
        self.video_client = VideoStreamingClient()
        
        # Cache for performance
        self._label_cache = {}
        self._frame_cache = {}
    
    async def initialize(self, group_name: str = None):
        """
        Initialize identification system - can be called without group for basic setup
        If group_name is provided, also loads the group model
        """
        try:
            if not self.is_initialized:
                logger.info("Initializing piece identification system (basic setup)")
                
                # Basic initialization without loading any specific group model
                # This prepares the processor but doesn't load detection system yet
                self.is_initialized = True
                
                # Initialize video client if needed
                if not hasattr(self.video_client, 'initialized') or not self.video_client.initialized:
                    # Basic video client setup
                    pass
                
                logger.info("Basic identification system initialized")
            
            # If group name is provided, load the group model
            if group_name and (not self.is_group_loaded or group_name != self.current_group_name):
                await self._load_group_model(group_name)
                
        except Exception as e:
            logger.error(f"Failed to initialize identification system: {e}")
            raise
    
    async def _load_group_model(self, group_name: str):
        """Load a specific group model"""
        try:
            logger.info(f"Loading group model for: {group_name}")
            
            loop = asyncio.get_event_loop()
            self.detection_system = await loop.run_in_executor(
                None, 
                lambda: IdentificationDetectionSystem(
                    confidence_threshold=self.identification_confidence_threshold,
                    group_name=group_name
                )
            )
            
            self.is_group_loaded = True
            self.current_group_name = group_name
            self.stats['device'] = str(self.detection_system.device)
            self.stats['current_group'] = group_name
            
            logger.info(f"Group model loaded on device: {self.detection_system.device} for group: {group_name}")
            
        except Exception as e:
            logger.error(f"Failed to load group model {group_name}: {e}")
            self.is_group_loaded = False
            self.current_group_name = None
            self.stats['current_group'] = None
            raise
    
    async def switch_group(self, group_name: str) -> bool:
        """
        Switch to a different group model
        Returns True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if group_name == self.current_group_name and self.is_group_loaded:
                logger.info(f"Already using group {group_name}")
                return True
            
            logger.info(f"Switching from group '{self.current_group_name}' to '{group_name}'")
            
            # Store previous group for potential rollback
            previous_group = self.current_group_name
            previous_detection_system = self.detection_system
            
            try:
                # Load new group model
                await self._load_group_model(group_name)
                
                # Clean up previous model if it was different
                if previous_detection_system and previous_group != group_name:
                    # Clean up previous model resources if needed
                    pass
                
                return True
                
            except Exception as e:
                # Rollback on failure
                logger.error(f"Failed to switch to group {group_name}, rolling back: {e}")
                self.current_group_name = previous_group
                self.detection_system = previous_detection_system
                self.is_group_loaded = previous_detection_system is not None
                raise
                
        except Exception as e:
            logger.error(f"Error switching to group {group_name}: {e}")
            return False
    
    def _check_group_loaded(self, operation_name: str):
        """Check if a group is loaded before performing identification operations"""
        if not self.is_initialized:
            raise Exception(f"Cannot perform {operation_name}: System not initialized. Call initialize() first.")
        
        if not self.is_group_loaded or not self.current_group_name:
            raise Exception(f"Cannot perform {operation_name}: No group loaded. Select a group first using switch_group().")
        
        if not self.detection_system:
            raise Exception(f"Cannot perform {operation_name}: Detection system not loaded.")
    
    async def identify_with_frame_capture(self, camera_id: int, group_name: str = None, 
                                        freeze_stream: bool = True, quality: int = 85) -> Dict[str, Any]:
        """
        Main identification method - capture frame and identify all pieces for specific group
        If group_name is provided, will switch to that group first
        If no group_name and no group is loaded, will raise an error
        """
        start_time = time.time()
        stream_frozen = False
        
        try:
            # Ensure system is initialized
            if not self.is_initialized:
                await self.initialize()
            
            # Handle group selection
            if group_name:
                if group_name != self.current_group_name:
                    success = await self.switch_group(group_name)
                    if not success:
                        raise Exception(f"Failed to switch to group {group_name}")
            
            # Check that we have a group loaded
            self._check_group_loaded("identification")
            
            logger.info(f"Starting identification for camera {camera_id}, group: {self.current_group_name}")
            
            # Freeze stream if requested
            if freeze_stream:
                freeze_success = await self.video_client.freeze_stream(camera_id)
                stream_frozen = freeze_success
            
            # Get current frame
            frame = await self.video_client.get_current_frame(camera_id)
            if frame is None:
                raise Exception(f"Could not get frame from camera {camera_id}")
            
            # Ensure frame is contiguous but don't force resize - let identification system handle it
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            logger.info(f"Running identification on frame from camera {camera_id} - Original size: {frame.shape}")
            
            # Perform identification using sliding window
            loop = asyncio.get_event_loop()
            identification_results = await loop.run_in_executor(
                None, 
                self.detection_system.identify_with_sliding_window,
                frame
            )
            
            # Parse identification results
            if isinstance(identification_results, tuple) and len(identification_results) >= 2:
                processed_frame = identification_results[0]
                detection_details = identification_results[1]
                
                logger.info(f"Identification completed - found {len(detection_details)} pieces")
            else:
                logger.warning(f"Unexpected identification results structure: {identification_results}")
                processed_frame = identification_results[0] if isinstance(identification_results, tuple) else identification_results
                detection_details = []
            
            # Get detailed results for comprehensive analysis
            detailed_results = await loop.run_in_executor(
                None,
                self.detection_system.get_detailed_identification_results,
                frame
            )
            
            # Create identification result structure
            identification_result = {
                'pieces': detailed_results,
                'detection_details': detection_details,  # Keep for compatibility
                'total_pieces_found': len(detailed_results),
                'analysis_timestamp': time.time(),
                'confidence_threshold_used': self.identification_confidence_threshold,
                'group_name': self.current_group_name
            }
            
            # Update frozen frame with results
            if processed_frame is not None and stream_frozen:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        await self.video_client.update_frozen_frame(camera_id, buffer.tobytes())
                except Exception as e:
                    logger.error(f"Error updating frozen frame: {e}")
            
            # Encode frame for response
            frame_b64 = ""
            if processed_frame is not None:
                try:
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                    success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                    if success:
                        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error encoding frame: {e}")
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_stats(identification_result, processing_time)
            
            # Generate summary
            summary = self.get_identification_summary(detailed_results)
            
            return {
                'success': True,
                'identification_result': identification_result,
                'identification_summary': summary,
                'frame_with_overlay': frame_b64,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': time.time(),
                'stream_frozen': stream_frozen,
                'group_name': self.current_group_name,
                'camera_id': camera_id
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error in identification: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'identification_result': {
                    'pieces': [], 
                    'detection_details': [],
                    'total_pieces_found': 0, 
                    'analysis_timestamp': time.time(),
                    'group_name': self.current_group_name
                },
                'identification_summary': self.get_identification_summary([]),
                'frame_with_overlay': "",
                'processing_time_ms': round(processing_time, 2),
                'timestamp': time.time(),
                'stream_frozen': stream_frozen,
                'group_name': self.current_group_name,
                'camera_id': camera_id
            }
    
    async def switch_group_and_identify(self, camera_id: int, new_group_name: str, 
                                      freeze_stream: bool = True, quality: int = 85) -> Dict[str, Any]:
        """
        Switch to a different group model and perform identification
        """
        try:
            # Ensure system is initialized
            if not self.is_initialized:
                await self.initialize()
            
            # Switch to new group
            success = await self.switch_group(new_group_name)
            if not success:
                raise Exception(f"Failed to switch to group {new_group_name}")
            
            # Perform identification with the new group
            return await self.identify_with_frame_capture(camera_id, None, freeze_stream, quality)
            
        except Exception as e:
            logger.error(f"Error switching group and identifying: {e}")
            return {
                'success': False,
                'error': str(e),
                'identification_result': {'pieces': [], 'total_pieces_found': 0, 'analysis_timestamp': time.time()},
                'identification_summary': self.get_identification_summary([]),
                'frame_with_overlay': "",
                'processing_time_ms': 0,
                'timestamp': time.time(),
                'stream_frozen': False,
                'group_name': new_group_name,
                'camera_id': camera_id
            }
    
    async def batch_identify_frames(self, camera_id: int, group_name: str = None, 
                                  num_frames: int = 5, interval_seconds: float = 1.0) -> Dict[str, Any]:
        """
        Identify pieces in multiple frames over time for better accuracy
        If group_name is provided, will switch to that group first
        """
        try:
            # Ensure system is initialized
            if not self.is_initialized:
                await self.initialize()
            
            # Handle group selection
            if group_name:
                if group_name != self.current_group_name:
                    success = await self.switch_group(group_name)
                    if not success:
                        raise Exception(f"Failed to switch to group {group_name}")
            
            # Check that we have a group loaded
            self._check_group_loaded("batch identification")
            
            all_results = []
            all_pieces = []
            
            for i in range(num_frames):
                logger.info(f"Capturing frame {i+1}/{num_frames} for batch identification")
                
                # Get frame without freezing stream (for continuous capture)
                frame = await self.video_client.get_current_frame(camera_id)
                if frame is None:
                    logger.warning(f"Could not get frame {i+1}, skipping")
                    continue
                
                # Ensure frame is contiguous
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                
                # Perform identification
                loop = asyncio.get_event_loop()
                identification_results = await loop.run_in_executor(
                    None, 
                    self.detection_system.identify_with_sliding_window,
                    frame
                )
                
                if isinstance(identification_results, tuple) and len(identification_results) >= 2:
                    detection_details = identification_results[1]
                    all_results.append({
                        'frame_index': i,
                        'pieces_found': len(detection_details),
                        'pieces': detection_details,
                        'timestamp': time.time()
                    })
                    all_pieces.extend(detection_details)
                
                # Wait between frames (except for the last one)
                if i < num_frames - 1:
                    await asyncio.sleep(interval_seconds)
            
            # Aggregate results
            unique_labels = set()
            total_pieces = len(all_pieces)
            confidence_scores = []
            
            for piece in all_pieces:
                unique_labels.add(piece['label'])
                confidence_scores.append(piece['confidence'])
            
            batch_summary = {
                'frames_processed': len(all_results),
                'total_pieces_found': total_pieces,
                'unique_labels_found': list(unique_labels),
                'average_pieces_per_frame': total_pieces / len(all_results) if all_results else 0,
                'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                'frame_results': all_results
            }
            
            return {
                'success': True,
                'batch_identification_result': batch_summary,
                'group_name': self.current_group_name,
                'camera_id': camera_id,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in batch identification: {e}")
            return {
                'success': False,
                'error': str(e),
                'batch_identification_result': None,
                'group_name': self.current_group_name,
                'camera_id': camera_id,
                'timestamp': time.time()
            }
    
    def get_identification_summary(self, pieces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate identification summary"""
        if not pieces:
            return {
                'total_pieces': 0,
                'unique_labels': [],
                'label_counts': {},
                'confidence_stats': {},
                'highest_confidence_piece': None,
                'lowest_confidence_piece': None,
                'pieces_by_confidence_level': {
                    'very_high': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0,
                    'very_low': 0
                }
            }
        
        # Count labels
        label_counts = {}
        confidences = []
        confidence_levels = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
        
        for piece in pieces:
            label = piece['label']
            confidence = piece['confidence']
            
            label_counts[label] = label_counts.get(label, 0) + 1
            confidences.append(confidence)
            
            # Count by confidence level
            conf_level = piece.get('confidence_level', 'unknown').lower().replace(' ', '_')
            if conf_level in confidence_levels:
                confidence_levels[conf_level] += 1
        
        # Find highest and lowest confidence pieces
        highest_confidence_piece = max(pieces, key=lambda p: p['confidence']) if pieces else None
        lowest_confidence_piece = min(pieces, key=lambda p: p['confidence']) if pieces else None
        
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
            } if highest_confidence_piece else None,
            'lowest_confidence_piece': {
                'label': lowest_confidence_piece['label'],
                'confidence': lowest_confidence_piece['confidence'],
                'confidence_percentage': lowest_confidence_piece['confidence_percentage']
            } if lowest_confidence_piece else None,
            'pieces_by_confidence_level': confidence_levels
        }
    
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
            'processing_time_ms': round(processing_time, 2),
            'group_name': identification_result.get('group_name', 'unknown')
        }
        
        self.stats['recent_identifications'].insert(0, recent_record)
        if len(self.stats['recent_identifications']) > 10:
            self.stats['recent_identifications'] = self.stats['recent_identifications'][:10]
    
    def set_confidence_threshold(self, threshold: float) -> bool:
        """Update confidence threshold"""
        if 0.1 <= threshold <= 1.0:
            self.identification_confidence_threshold = threshold
            self.stats['confidence_threshold'] = threshold
            
            if self.detection_system:
                self.detection_system.set_confidence_threshold(threshold)
            
            logger.info(f"Confidence threshold updated to {threshold}")
            return True
        return False
    
    def get_available_groups(self) -> List[str]:
        """Get list of available group models"""
        try:
            # This would need to be implemented based on your model storage structure
            # For now, return common group patterns
            return ['E539', 'G053', 'A123', 'B456']  # Example groups
        except Exception as e:
            logger.error(f"Error getting available groups: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        if not self.detection_system:
            return {
                'is_loaded': False,
                'current_group': self.current_group_name,
                'error': 'Model not initialized'
            }
        
        try:
            model_info = self.detection_system.get_model_info()
            return {
                'is_loaded': True,
                'current_group': self.current_group_name,
                **model_info
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'is_loaded': False,
                'current_group': self.current_group_name,
                'error': str(e)
            }
    
    async def unfreeze_stream(self, camera_id: int) -> bool:
        """Unfreeze video stream"""
        return await self.video_client.unfreeze_stream(camera_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get identification statistics"""
        return {
            'is_initialized': self.is_initialized,
            'is_group_loaded': self.is_group_loaded,
            'current_group_name': self.current_group_name,
            **self.stats
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.video_client.close()

# Global identification processor instance
piece_identification_processor = PieceIdentificationProcessor()