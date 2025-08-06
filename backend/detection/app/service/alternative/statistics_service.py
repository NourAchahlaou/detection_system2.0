# statistics_service.py - Enhanced Detection Statistics Service

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, and_, desc, asc
from datetime import datetime, timedelta
import json

from detection.app.db.models.detectionLot import DetectionLot
from detection.app.db.models.detectionSession import DetectionSession
from detection.app.db.session import get_session

logger = logging.getLogger(__name__)

@dataclass
class PieceStatistics:
    """Statistics for detected pieces"""
    piece_label: str
    piece_id: int
    total_detected: int
    correct_count: int
    misplaced_count: int
    accuracy_percentage: float
    last_detection_time: Optional[str] = None

@dataclass
class LotProgress:
    """Progress information for a detection lot"""
    lot_id: int
    lot_name: str
    expected_piece_label: str
    expected_piece_id: int
    expected_piece_number: int
    current_detected: int
    correct_pieces: int
    misplaced_pieces: int
    completion_percentage: float
    accuracy_percentage: float
    is_completed: bool
    created_at: str
    completed_at: Optional[str] = None
    total_sessions: int = 0

@dataclass
class DetectionOverview:
    """Overall detection statistics"""
    total_lots: int
    active_lots: int
    completed_lots: int
    total_sessions: int
    total_pieces_detected: int
    overall_accuracy: float
    average_session_time: float
    most_detected_piece: Optional[str] = None
    most_accurate_piece: Optional[str] = None

@dataclass
class RealTimeStats:
    """Real-time detection statistics"""
    current_lot: Optional[LotProgress] = None
    recent_detections: List[Dict[str, Any]] = None
    active_cameras: List[int] = None
    detection_rate_last_hour: float = 0.0
    system_performance: Dict[str, Any] = None

class DetectionStatisticsService:
    """Service for generating detection statistics and analytics"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 30  # 30 seconds cache
        self.last_cache_update = {}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is valid for given key"""
        if key not in self.last_cache_update:
            return False
        return (time.time() - self.last_cache_update[key]) < self.cache_timeout
    
    def _update_cache(self, key: str, data: Any):
        """Update cache with new data"""
        self.cache[key] = data
        self.last_cache_update[key] = time.time()
    
    def get_lot_progress(self, lot_id: int, db: Session) -> Optional[LotProgress]:
        """Get detailed progress for a specific lot"""
        try:
            cache_key = f"lot_progress_{lot_id}"
            
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get lot with sessions
            lot = db.query(DetectionLot).options(
                selectinload(DetectionLot.detection_sessions)
            ).filter(DetectionLot.id == lot_id).first()
            
            if not lot:
                return None
            
            # Calculate statistics from sessions
            total_sessions = len(lot.detection_sessions)
            correct_pieces = sum(session.correct_pieces_count for session in lot.detection_sessions)
            misplaced_pieces = sum(session.misplaced_pieces_count for session in lot.detection_sessions)
            current_detected = correct_pieces + misplaced_pieces
            
            # Calculate percentages
            completion_percentage = min((current_detected / lot.expected_piece_number) * 100, 100) if lot.expected_piece_number > 0 else 0
            accuracy_percentage = (correct_pieces / current_detected * 100) if current_detected > 0 else 0
            
            progress = LotProgress(
                lot_id=lot.id,
                lot_name=lot.lot_name,
                expected_piece_label=f"Piece_{lot.expected_piece_id}",  # You might want to fetch actual label
                expected_piece_id=lot.expected_piece_id,
                expected_piece_number=lot.expected_piece_number,
                current_detected=current_detected,
                correct_pieces=correct_pieces,
                misplaced_pieces=misplaced_pieces,
                completion_percentage=round(completion_percentage, 1),
                accuracy_percentage=round(accuracy_percentage, 1),
                is_completed=lot.is_target_match,
                created_at=lot.created_at.isoformat() if lot.created_at else datetime.utcnow().isoformat(),
                completed_at=lot.completed_at.isoformat() if lot.completed_at else None,
                total_sessions=total_sessions
            )
            
            self._update_cache(cache_key, progress)
            return progress
            
        except Exception as e:
            logger.error(f"❌ Error getting lot progress for lot {lot_id}: {e}")
            return None
    
    def get_piece_statistics(self, piece_id: Optional[int] = None, db: Session = None) -> List[PieceStatistics]:
        """Get statistics for detected pieces"""
        try:
            cache_key = f"piece_stats_{piece_id or 'all'}"
            
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Query to get piece statistics
            query = db.query(
                DetectionLot.expected_piece_id,
                func.count(DetectionSession.id).label('total_sessions'),
                func.sum(DetectionSession.correct_pieces_count).label('total_correct'),
                func.sum(DetectionSession.misplaced_pieces_count).label('total_misplaced'),
                func.max(DetectionSession.created_at).label('last_detection')
            ).join(
                DetectionSession, DetectionLot.id == DetectionSession.lot_id
            ).group_by(DetectionLot.expected_piece_id)
            
            if piece_id:
                query = query.filter(DetectionLot.expected_piece_id == piece_id)
            
            results = query.all()
            
            piece_stats = []
            for result in results:
                total_detected = (result.total_correct or 0) + (result.total_misplaced or 0)
                accuracy = ((result.total_correct or 0) / total_detected * 100) if total_detected > 0 else 0
                
                piece_stats.append(PieceStatistics(
                    piece_label=f"Piece_{result.expected_piece_id}",
                    piece_id=result.expected_piece_id,
                    total_detected=total_detected,
                    correct_count=result.total_correct or 0,
                    misplaced_count=result.total_misplaced or 0,
                    accuracy_percentage=round(accuracy, 1),
                    last_detection_time=result.last_detection.isoformat() if result.last_detection else None
                ))
            
            self._update_cache(cache_key, piece_stats)
            return piece_stats
            
        except Exception as e:
            logger.error(f"❌ Error getting piece statistics: {e}")
            return []
    
    def get_detection_overview(self, db: Session) -> DetectionOverview:
        """Get overall detection system overview"""
        try:
            cache_key = "detection_overview"
            
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get lot statistics
            total_lots = db.query(DetectionLot).count()
            completed_lots = db.query(DetectionLot).filter(DetectionLot.is_target_match == True).count()
            active_lots = total_lots - completed_lots
            
            # Get session statistics
            total_sessions = db.query(DetectionSession).count()
            
            # Calculate overall accuracy
            accuracy_result = db.query(
                func.sum(DetectionSession.correct_pieces_count).label('total_correct'),
                func.sum(DetectionSession.misplaced_pieces_count).label('total_misplaced')
            ).first()
            
            total_correct = accuracy_result.total_correct or 0
            total_misplaced = accuracy_result.total_misplaced or 0
            total_pieces = total_correct + total_misplaced
            overall_accuracy = (total_correct / total_pieces * 100) if total_pieces > 0 else 0
            
            # Get most detected piece
            most_detected = db.query(
                DetectionLot.expected_piece_id,
                func.count(DetectionSession.id).label('session_count')
            ).join(
                DetectionSession, DetectionLot.id == DetectionSession.lot_id
            ).group_by(DetectionLot.expected_piece_id).order_by(
                func.count(DetectionSession.id).desc()
            ).first()
            
            # Get most accurate piece
            most_accurate = db.query(
                DetectionLot.expected_piece_id,
                (func.sum(DetectionSession.correct_pieces_count) / 
                 func.sum(DetectionSession.correct_pieces_count + DetectionSession.misplaced_pieces_count) * 100).label('accuracy')
            ).join(
                DetectionSession, DetectionLot.id == DetectionSession.lot_id
            ).group_by(DetectionLot.expected_piece_id).having(
                func.sum(DetectionSession.correct_pieces_count + DetectionSession.misplaced_pieces_count) > 0
            ).order_by(desc('accuracy')).first()
            
            overview = DetectionOverview(
                total_lots=total_lots,
                active_lots=active_lots,
                completed_lots=completed_lots,
                total_sessions=total_sessions,
                total_pieces_detected=total_pieces,
                overall_accuracy=round(overall_accuracy, 1),
                average_session_time=0.0,  # Can be calculated if you store session duration
                most_detected_piece=f"Piece_{most_detected.expected_piece_id}" if most_detected else None,
                most_accurate_piece=f"Piece_{most_accurate.expected_piece_id}" if most_accurate else None
            )
            
            self._update_cache(cache_key, overview)
            return overview
            
        except Exception as e:
            logger.error(f"❌ Error getting detection overview: {e}")
            return DetectionOverview(
                total_lots=0, active_lots=0, completed_lots=0,
                total_sessions=0, total_pieces_detected=0,
                overall_accuracy=0.0, average_session_time=0.0
            )
    
    def get_real_time_stats(self, camera_id: Optional[int] = None, db: Session = None) -> RealTimeStats:
        """Get real-time detection statistics"""
        try:
            cache_key = f"realtime_stats_{camera_id or 'all'}"
            
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]
            
            # Get current active lot (most recent non-completed lot)
            current_lot_data = db.query(DetectionLot).filter(
                DetectionLot.is_target_match == False
            ).order_by(DetectionLot.created_at.desc()).first()
            
            current_lot = None
            if current_lot_data:
                current_lot = self.get_lot_progress(current_lot_data.id, db)
            
            # Get recent detections (last 10)
            recent_sessions = db.query(DetectionSession).options(
                selectinload(DetectionSession.detection_lot)
            ).order_by(DetectionSession.created_at.desc()).limit(10).all()
            
            recent_detections = []
            for session in recent_sessions:
                recent_detections.append({
                    'session_id': session.id,
                    'lot_name': session.detection_lot.lot_name,
                    'expected_piece_id': session.detection_lot.expected_piece_id,
                    'correct_pieces': session.correct_pieces_count,
                    'misplaced_pieces': session.misplaced_pieces_count,
                    'is_target_match': session.is_target_match,
                    'confidence': session.confidence_score,
                    'created_at': session.created_at.isoformat() if session.created_at else None
                })
            
            # Calculate detection rate in last hour
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_detection_count = db.query(DetectionSession).filter(
                DetectionSession.created_at >= one_hour_ago
            ).count()
            
            stats = RealTimeStats(
                current_lot=current_lot,
                recent_detections=recent_detections,
                active_cameras=[camera_id] if camera_id else [],
                detection_rate_last_hour=recent_detection_count / 60.0,  # detections per minute
                system_performance={
                    'total_sessions_today': self._get_sessions_today(db),
                    'accuracy_today': self._get_accuracy_today(db),
                    'avg_processing_time': 0.0  # Can be calculated if stored
                }
            )
            
            self._update_cache(cache_key, stats)
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting real-time stats: {e}")
            return RealTimeStats()
    
    def _get_sessions_today(self, db: Session) -> int:
        """Get number of sessions today"""
        try:
            today = datetime.utcnow().date()
            return db.query(DetectionSession).filter(
                func.date(DetectionSession.created_at) == today
            ).count()
        except:
            return 0
    
    def _get_accuracy_today(self, db: Session) -> float:
        """Get accuracy percentage for today"""
        try:
            today = datetime.utcnow().date()
            result = db.query(
                func.sum(DetectionSession.correct_pieces_count).label('correct'),
                func.sum(DetectionSession.misplaced_pieces_count).label('misplaced')
            ).filter(
                func.date(DetectionSession.created_at) == today
            ).first()
            
            if result and (result.correct or result.misplaced):
                total = (result.correct or 0) + (result.misplaced or 0)
                return round((result.correct or 0) / total * 100, 1)
            return 0.0
        except:
            return 0.0
    
    def get_lot_completion_timeline(self, lot_id: int, db: Session) -> List[Dict[str, Any]]:
        """Get timeline of detections for a specific lot"""
        try:
            sessions = db.query(DetectionSession).filter(
                DetectionSession.lot_id == lot_id
            ).order_by(DetectionSession.created_at.asc()).all()
            
            timeline = []
            cumulative_correct = 0
            cumulative_total = 0
            
            for session in sessions:
                cumulative_correct += session.correct_pieces_count
                cumulative_total += session.correct_pieces_count + session.misplaced_pieces_count
                
                timeline.append({
                    'session_id': session.id,
                    'timestamp': session.created_at.isoformat() if session.created_at else None,
                    'correct_pieces': session.correct_pieces_count,
                    'misplaced_pieces': session.misplaced_pieces_count,
                    'cumulative_correct': cumulative_correct,
                    'cumulative_total': cumulative_total,
                    'cumulative_accuracy': (cumulative_correct / cumulative_total * 100) if cumulative_total > 0 else 0,
                    'is_target_match': session.is_target_match,
                    'confidence': session.confidence_score
                })
            
            return timeline
            
        except Exception as e:
            logger.error(f"❌ Error getting lot timeline: {e}")
            return []
    
    def get_performance_analytics(self, db: Session, days: int = 7) -> Dict[str, Any]:
        """Get performance analytics for the last N days"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Daily statistics
            daily_stats = db.query(
                func.date(DetectionSession.created_at).label('date'),
                func.count(DetectionSession.id).label('sessions'),
                func.sum(DetectionSession.correct_pieces_count).label('correct'),
                func.sum(DetectionSession.misplaced_pieces_count).label('misplaced'),
                func.avg(DetectionSession.confidence_score).label('avg_confidence')
            ).filter(
                DetectionSession.created_at >= start_date
            ).group_by(
                func.date(DetectionSession.created_at)
            ).order_by('date').all()
            
            analytics = {
                'period_days': days,
                'daily_performance': [],
                'total_sessions': sum(stat.sessions for stat in daily_stats),
                'total_correct': sum(stat.correct or 0 for stat in daily_stats),
                'total_misplaced': sum(stat.misplaced or 0 for stat in daily_stats),
                'average_confidence': 0.0,
                'trend_analysis': {
                    'accuracy_trend': 'stable',
                    'volume_trend': 'stable',
                    'confidence_trend': 'stable'
                }
            }
            
            for stat in daily_stats:
                total_pieces = (stat.correct or 0) + (stat.misplaced or 0)
                accuracy = (stat.correct or 0) / total_pieces * 100 if total_pieces > 0 else 0
                
                analytics['daily_performance'].append({
                    'date': stat.date.isoformat() if stat.date else None,
                    'sessions': stat.sessions,
                    'correct_pieces': stat.correct or 0,
                    'misplaced_pieces': stat.misplaced or 0,
                    'accuracy_percentage': round(accuracy, 1),
                    'average_confidence': round(stat.avg_confidence or 0, 2)
                })
            
            # Calculate average confidence
            if daily_stats:
                analytics['average_confidence'] = round(
                    sum(stat.avg_confidence or 0 for stat in daily_stats) / len(daily_stats), 2
                )
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Error getting performance analytics: {e}")
            return {
                'period_days': days,
                'daily_performance': [],
                'total_sessions': 0,
                'total_correct': 0,
                'total_misplaced': 0,
                'average_confidence': 0.0,
                'trend_analysis': {
                    'accuracy_trend': 'unknown',
                    'volume_trend': 'unknown', 
                    'confidence_trend': 'unknown'
                }
            }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.last_cache_update.clear()
        logger.info("📊 Statistics cache cleared")

# Global statistics service instance
detection_statistics_service = DetectionStatisticsService()