# lot_dashboard_service.py - Fixed backend service for lot session dashboard data

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, and_, case
import re

from detection.app.db.models.detectionLot import DetectionLot
from detection.app.db.models.detectionSession import DetectionSession
from detection.app.db.models.piece import Piece

logger = logging.getLogger(__name__)

class LotSessionDashboardService:
    """Service for providing lot session dashboard data grouped by piece groups"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def _ensure_timezone_aware(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC if naive)"""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    
    def _get_utc_now(self) -> datetime:
        """Get current UTC datetime that's timezone-aware"""
        return datetime.now(timezone.utc)
        
    def _extract_piece_group(self, piecelabel: str) -> str:
        """
        Extract piece group from lot name
        Examples: 
        - G123.12345.123.12 -> G123
        """
        if '.' in piecelabel:
            # Handle format like G123.12345.123.12
            return piecelabel.split('.')[0]
        else:
            # Fallback to first 4 characters
            return piecelabel[:4].upper() if len(piecelabel) > 4 else piecelabel.upper()
    
    def _get_piece_by_id(self, db: Session, piece_id: int) -> Piece:
        """Get a piece by its ID."""
        piece = db.query(Piece).filter(Piece.id == piece_id).first()
        if not piece:
            raise HTTPException(status_code=404, detail="Piece not found")
        return piece
    
    def _get_lot_status(self, lot: DetectionLot) -> str:
        """Determine lot status based on lot and sessions data"""
        if not lot.detection_sessions:
            return 'pending'
        
        # Check if lot is marked as completed
        if lot.completed_at:
            return 'completed' if lot.is_target_match else 'failed'
        
        # Check if any session is currently running (based on recent activity)
        latest_session = max(lot.detection_sessions, key=lambda s: self._ensure_timezone_aware(s.created_at))
        latest_session_time = self._ensure_timezone_aware(latest_session.created_at)
        time_since_last = self._get_utc_now() - latest_session_time
        
        if time_since_last < timedelta(minutes=5):
            return 'running'
        
        # Check overall lot success
        if lot.is_target_match:
            return 'completed'
        
        # Check if there are failed attempts
        failed_sessions = [s for s in lot.detection_sessions if not s.is_target_match]
        if failed_sessions:
            return 'failed'
            
        return 'in_progress'
    
    def _calculate_lot_match_stats(self, lot: DetectionLot) -> Dict[str, Any]:
        """Calculate lot matching statistics"""
        sessions = lot.detection_sessions
        total_sessions = len(sessions)
        
        if total_sessions == 0:
            return {
                'totalSessions': 0,
                'successfulSessions': 0,
                'failedSessions': 0,
                'sessionSuccessRate': 0.0,
                'isLotMatched': False,
                'lotMatchConfidence': 0.0
            }
        
        successful_sessions = [s for s in sessions if s.is_target_match]
        failed_sessions = [s for s in sessions if not s.is_target_match]
        
        session_success_rate = (len(successful_sessions) / total_sessions) * 100
        
        # Calculate average confidence for successful sessions
        lot_match_confidence = 0.0
        if successful_sessions:
            lot_match_confidence = (sum(s.confidence_score for s in successful_sessions) / len(successful_sessions)) * 100
        
        return {
            'totalSessions': total_sessions,
            'successfulSessions': len(successful_sessions),
            'failedSessions': len(failed_sessions),
            'sessionSuccessRate': round(session_success_rate, 1),
            'isLotMatched': lot.is_target_match,
            'lotMatchConfidence': round(lot_match_confidence, 1),
            'avgProcessingTime': self._calculate_avg_processing_time(sessions),
            'avgDetectionRate': self._calculate_avg_detection_rate(sessions)
        }
    
    def _calculate_avg_processing_time(self, sessions: List) -> float:
        """Calculate average processing time for sessions"""
        if not sessions:
            return 0.0
        # Simulate processing time based on session ID (replace with actual field if available)
        avg_time = sum(500 + (session.id % 2000) for session in sessions) / len(sessions)
        return round(avg_time, 1)
    
    def _calculate_avg_detection_rate(self, sessions: List) -> float:
        """Calculate average detection rate for sessions"""
        if not sessions:
            return 0.0
        detection_rates = [s.detection_rate for s in sessions if s.detection_rate is not None]
        if not detection_rates:
            return 0.0
        return round(sum(detection_rates) / len(detection_rates), 1)
    
    def get_dashboard_data(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive dashboard data grouped by piece groups"""
        try:
            # Get all lots with their sessions and piece information
            lots_query = (
                db.query(DetectionLot)
                .options(joinedload(DetectionLot.detection_sessions))
                .order_by(desc(DetectionLot.created_at))
            )
            
            lots = lots_query.all()
            
            # Get piece information
            pieces = db.query(Piece).all()
            piece_lookup = {piece.id: piece.piece_label for piece in pieces}
            
            # Process lots and group by piece groups
            grouped_lots = {}
            
            for lot in lots:
                # Get piece object by ID
                piece_object = self._get_piece_by_id(db, lot.expected_piece_id)
                piece_label = piece_object.piece_label

                # Extract piece group from piece label (not lot name)
                piece_group = self._extract_piece_group(piece_label)
                
                # Calculate lot matching statistics
                lot_stats = self._calculate_lot_match_stats(lot)
                
                # Process sessions for this lot
                sessions_data = []
                for idx, session in enumerate(lot.detection_sessions):
                    # Determine detected piece (simulate based on match status)
                    if session.is_target_match:
                        detected_piece = piece_label
                    else:
                        # Simulate misdetection with other pieces
                        other_pieces = [p for p in piece_lookup.values() if p != piece_label]
                        detected_piece = other_pieces[0] if other_pieces else "unknown"
                    
                    session_created_at = self._ensure_timezone_aware(session.created_at)
                    
                    session_data = {
                        'id': f"session_{lot.id}_{idx + 1}",
                        'sessionNumber': idx + 1,
                        'cameraId': (session.id % 4) + 1,  # Simulate camera assignment
                        'targetPiece': piece_label,
                        'detectedPiece': detected_piece,
                        'confidence': round(session.confidence_score * 100, 1),
                        'isTargetMatch': session.is_target_match,
                        'status': 'completed',  # All stored sessions are completed
                        'timestamp': session_created_at.isoformat(),
                        'processingTime': int(500 + (session.id % 2000)),  # Simulate processing time
                        'detectionRate': session.detection_rate,
                        'correctPiecesCount': getattr(session, 'correct_pieces_count', 0),
                        'misplacedPiecesCount': getattr(session, 'misplaced_pieces_count', 0),
                        'totalPiecesDetected': getattr(session, 'total_pieces_detected', 1)
                    }
                    sessions_data.append(session_data)
                
                # Calculate lot timestamps
                lot_created_at = self._ensure_timezone_aware(lot.created_at)
                lot_completed_at = self._ensure_timezone_aware(lot.completed_at) if lot.completed_at else None
                
                # Calculate last activity timestamp
                session_timestamps = [self._ensure_timezone_aware(s.created_at) for s in lot.detection_sessions]
                all_timestamps = [lot_created_at] + session_timestamps
                last_activity = max(all_timestamps)
                
                lot_data = {
                    'id': lot.id,
                    'group': piece_group,
                    'lotName': lot.lot_name,
                    'expectedPiece': piece_label,
                    'expectedPieceNumber': lot.expected_piece_number,
                    'status': self._get_lot_status(lot),
                    'sessions': sessions_data,
                    'createdAt': lot_created_at.isoformat(),
                    'completedAt': lot_completed_at.isoformat() if lot_completed_at else None,
                    'lastActivity': last_activity.isoformat(),
                    # Lot matching statistics
                    **lot_stats
                }
                
                # Group by piece group
                if piece_group not in grouped_lots:
                    grouped_lots[piece_group] = []
                grouped_lots[piece_group].append(lot_data)
            
            # Calculate group statistics
            group_stats = {}
            for group_name, group_lots in grouped_lots.items():
                total_lots = len(group_lots)
                matched_lots = len([lot for lot in group_lots if lot['isLotMatched']])
                total_sessions = sum(lot['totalSessions'] for lot in group_lots)
                total_successful_sessions = sum(lot['successfulSessions'] for lot in group_lots)
                
                # Calculate averages
                avg_session_success_rate = (
                    sum(lot['sessionSuccessRate'] for lot in group_lots) / total_lots
                ) if total_lots > 0 else 0
                
                avg_lot_confidence = (
                    sum(lot['lotMatchConfidence'] for lot in group_lots if lot['lotMatchConfidence'] > 0) /
                    len([lot for lot in group_lots if lot['lotMatchConfidence'] > 0])
                ) if any(lot['lotMatchConfidence'] > 0 for lot in group_lots) else 0
                
                last_activity_times = [
                    datetime.fromisoformat(lot['lastActivity'].replace('Z', '+00:00')) 
                    for lot in group_lots
                ]
                last_activity = max(last_activity_times) if last_activity_times else self._get_utc_now()
                
                group_stats[group_name] = {
                    'groupName': group_name,
                    'totalLots': total_lots,
                    'matchedLots': matched_lots,
                    'lotMatchRate': round((matched_lots / total_lots * 100), 1) if total_lots > 0 else 0,
                    'totalSessions': total_sessions,
                    'successfulSessions': total_successful_sessions,
                    'avgSessionSuccessRate': round(avg_session_success_rate, 1),
                    'avgLotConfidence': round(avg_lot_confidence, 1),
                    'lastActivity': last_activity.isoformat(),
                    'activeLots': len([lot for lot in group_lots if lot['status'] in ['running', 'in_progress']]),
                    'completedLots': len([lot for lot in group_lots if lot['status'] == 'completed']),
                    'failedLots': len([lot for lot in group_lots if lot['status'] == 'failed'])
                }
            
            # Calculate overall statistics
            all_lots = [lot for group_lots in grouped_lots.values() for lot in group_lots]
            total_lots = len(all_lots)
            matched_lots = len([lot for lot in all_lots if lot['isLotMatched']])
            total_sessions = sum(lot['totalSessions'] for lot in all_lots)
            successful_sessions = sum(lot['successfulSessions'] for lot in all_lots)
            
            statistics = {
                'totalGroups': len(grouped_lots),
                'totalLots': total_lots,
                'matchedLots': matched_lots,
                'lotMatchRate': round((matched_lots / total_lots * 100), 1) if total_lots > 0 else 0,
                'totalSessions': total_sessions,
                'successfulSessions': successful_sessions,
                'sessionSuccessRate': round((successful_sessions / total_sessions * 100), 1) if total_sessions > 0 else 0,
                'avgLotConfidence': round(
                    sum(lot['lotMatchConfidence'] for lot in all_lots if lot['lotMatchConfidence'] > 0) /
                    len([lot for lot in all_lots if lot['lotMatchConfidence'] > 0])
                , 1) if any(lot['lotMatchConfidence'] > 0 for lot in all_lots) else 0,
                'activeGroups': len([
                    group for group, lots in grouped_lots.items()
                    if any(lot['status'] in ['running', 'in_progress'] for lot in lots)
                ]),
                'completedGroups': len([
                    group for group, lots in grouped_lots.items()
                    if all(lot['status'] in ['completed', 'failed'] for lot in lots)
                ])
            }
            
            return {
                'success': True,
                'statistics': statistics,
                'groupedLots': grouped_lots,
                'groupStats': group_stats,
                'timestamp': self._get_utc_now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            raise
    
    def get_lot_details(self, lot_id: int, db: Session) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific lot"""
        try:
            lot = (
                db.query(DetectionLot)
                .options(joinedload(DetectionLot.detection_sessions))
                .filter(DetectionLot.id == lot_id)
                .first()
            )
            
            if not lot:
                return None
            
            # Get piece information
            piece = db.query(Piece).filter(Piece.id == lot.expected_piece_id).first()
            piece_label = piece.piece_label if piece else f"piece_{lot.expected_piece_id}"
            
            # Calculate lot matching statistics
            lot_stats = self._calculate_lot_match_stats(lot)
            
            # Process sessions
            sessions_data = []
            for idx, session in enumerate(lot.detection_sessions):
                detected_piece = piece_label if session.is_target_match else "unknown"
                session_created_at = self._ensure_timezone_aware(session.created_at)
                
                session_data = {
                    'id': f"session_{lot.id}_{idx + 1}",
                    'sessionNumber': idx + 1,
                    'cameraId': (session.id % 4) + 1,
                    'targetPiece': piece_label,
                    'detectedPiece': detected_piece,
                    'confidence': round(session.confidence_score * 100, 1),
                    'isTargetMatch': session.is_target_match,
                    'status': 'completed',
                    'timestamp': session_created_at.isoformat(),
                    'processingTime': int(500 + (session.id % 2000)),
                    'detectionRate': session.detection_rate,
                    'correctPiecesCount': getattr(session, 'correct_pieces_count', 0),
                    'misplacedPiecesCount': getattr(session, 'misplaced_pieces_count', 0),
                    'totalPiecesDetected': getattr(session, 'total_pieces_detected', 1)
                }
                sessions_data.append(session_data)
            
            # Calculate timestamps
            lot_created_at = self._ensure_timezone_aware(lot.created_at)
            lot_completed_at = self._ensure_timezone_aware(lot.completed_at) if lot.completed_at else None
            
            session_timestamps = [self._ensure_timezone_aware(s.created_at) for s in lot.detection_sessions]
            all_timestamps = [lot_created_at] + session_timestamps
            last_activity = max(all_timestamps)
            
            return {
                'id': lot.id,
                'lotName': lot.lot_name,
                'expectedPiece': piece_label,
                'expectedPieceNumber': lot.expected_piece_number,
                'group': self._extract_piece_group(piece_label),  # Fixed: use piece_label instead of lot.lot_name
                'status': self._get_lot_status(lot),
                'sessions': sessions_data,
                'createdAt': lot_created_at.isoformat(),
                'completedAt': lot_completed_at.isoformat() if lot_completed_at else None,
                'lastActivity': last_activity.isoformat(),
                # Lot matching statistics
                **lot_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting lot details for {lot_id}: {e}")
            raise
    
    def get_group_summary(self, group_name: str, db: Session) -> Dict[str, Any]:
        """Get summary for a specific piece group"""
        try:
            # Get all lots with their sessions and piece information
            lots = db.query(DetectionLot).options(joinedload(DetectionLot.detection_sessions)).all()
            
            # Get piece information for filtering
            pieces = db.query(Piece).all()
            piece_lookup = {piece.id: piece.piece_label for piece in pieces}
            
            group_lots = []
            for lot in lots:
                piece_label = piece_lookup.get(lot.expected_piece_id, f"piece_{lot.expected_piece_id}")
                lot_group = self._extract_piece_group(piece_label)
                if lot_group == group_name:
                    group_lots.append(lot)
            
            if not group_lots:
                return {'success': False, 'message': f'No lots found for group {group_name}'}
            
            # Calculate group statistics
            total_lots = len(group_lots)
            matched_lots = sum(1 for lot in group_lots if lot.is_target_match)
            total_sessions = sum(len(lot.detection_sessions) for lot in group_lots)
            total_successful_sessions = sum(
                len([s for s in lot.detection_sessions if s.is_target_match]) 
                for lot in group_lots
            )
            
            # Calculate rates
            lot_match_rate = (matched_lots / total_lots * 100) if total_lots > 0 else 0
            session_success_rate = (total_successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
            
            # Calculate confidence scores
            confidence_scores = []
            for lot in group_lots:
                successful_sessions = [s for s in lot.detection_sessions if s.is_target_match]
                if successful_sessions:
                    avg_confidence = sum(s.confidence_score for s in successful_sessions) / len(successful_sessions) * 100
                    confidence_scores.append(avg_confidence)
            
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Calculate last activity
            all_timestamps = []
            for lot in group_lots:
                lot_created = self._ensure_timezone_aware(lot.created_at)
                all_timestamps.append(lot_created)
                for session in lot.detection_sessions:
                    session_created = self._ensure_timezone_aware(session.created_at)
                    all_timestamps.append(session_created)
            
            last_activity = max(all_timestamps) if all_timestamps else self._get_utc_now()
            
            # Count lots by status
            status_counts = {
                'completed': 0,
                'failed': 0,
                'running': 0,
                'in_progress': 0,
                'pending': 0
            }
            
            for lot in group_lots:
                status = self._get_lot_status(lot)
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'success': True,
                'groupName': group_name,
                'totalLots': total_lots,
                'matchedLots': matched_lots,
                'lotMatchRate': round(lot_match_rate, 1),
                'totalSessions': total_sessions,
                'successfulSessions': total_successful_sessions,
                'sessionSuccessRate': round(session_success_rate, 1),
                'avgConfidence': round(avg_confidence, 1),
                'lastActivity': last_activity.isoformat(),
                'statusBreakdown': status_counts,
                'completedLots': status_counts['completed'],
                'activeLots': status_counts['running'] + status_counts['in_progress']
            }
            
        except Exception as e:
            logger.error(f"Error getting group summary for {group_name}: {e}")
            raise

# Global service instance
lot_dashboard_service = LotSessionDashboardService()