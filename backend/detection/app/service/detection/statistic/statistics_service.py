# lot_statistics.py - Comprehensive lot and session statistics and analytics

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import Integer, func, and_
from dataclasses import dataclass
from enum import Enum

# Import models
from detection.app.db.models.detectionLot import DetectionLot
from detection.app.db.models.detectionSession import DetectionSession
from detection.app.db.models.detectedPiece import DetectedPiece

logger = logging.getLogger(__name__)

class LotStatus(Enum):
    SUCCESSFUL_FIRST_TRY = "successful_first_try"  # Matched on first session
    MIXED_CORRECTED = "mixed_corrected"           # Was wrong but later corrected
    MIXED_PENDING = "mixed_pending"               # Still has issues
    NO_SESSIONS = "no_sessions"                   # No detection sessions yet

@dataclass
class LotStatsSummary:
    total_lots: int
    completed_lots: int
    pending_lots: int
    successful_first_try: int
    mixed_corrected: int
    mixed_pending: int
    avg_sessions_to_complete: float
    completion_rate: float
    first_try_success_rate: float

@dataclass
class LotTimeStats:
    period: str
    lots_created: int
    lots_completed: int
    successful_first_try: int
    mixed_lots: int
    completion_rate: float
    first_try_success_rate: float

@dataclass
class SessionAnalytics:
    total_sessions: int
    successful_sessions: int
    failed_sessions: int
    avg_confidence: float
    avg_pieces_detected: float
    avg_processing_time: float
    success_rate: float

@dataclass
class MixUpAnalysis:
    expected_piece_id: int
    expected_piece_label: str
    total_mixed_lots: int
    most_common_wrong_pieces: List[Dict[str, Any]]
    avg_sessions_to_fix: float
    mix_up_rate: float

class LotStatisticsService:
    """Service for generating comprehensive lot and session statistics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_overall_lot_summary(self, db: Session) -> LotStatsSummary:
        """Get overall summary of all lots and their status"""
        try:
            # Basic lot counts
            total_lots = db.query(DetectionLot).count()
            completed_lots = db.query(DetectionLot).filter(DetectionLot.is_target_match == True).count()
            pending_lots = total_lots - completed_lots
            
            # Analyze lot success patterns
            lot_analysis = self._analyze_lot_success_patterns(db)
            
            successful_first_try = lot_analysis['successful_first_try']
            mixed_corrected = lot_analysis['mixed_corrected']  
            mixed_pending = lot_analysis['mixed_pending']
            
            # Calculate completion metrics
            completion_rate = (completed_lots / total_lots * 100) if total_lots > 0 else 0
            first_try_success_rate = (successful_first_try / total_lots * 100) if total_lots > 0 else 0
            
            # Average sessions to complete (only for completed lots)
            session_count = (
                db.query(
                    DetectionLot.id,
                    func.count(DetectionSession.id).label('session_count')
                )
                .join(DetectionSession)
                .filter(DetectionLot.is_target_match == True)
                .group_by(DetectionLot.id)
                .subquery('session_count')
            )
            avg_sessions_query = (
                db.query(func.avg(session_count.c.session_count))
                .select_from(session_count)
            )
            avg_sessions_to_complete = avg_sessions_query.scalar() or 0
            
            return LotStatsSummary(
                total_lots=total_lots,
                completed_lots=completed_lots,
                pending_lots=pending_lots,
                successful_first_try=successful_first_try,
                mixed_corrected=mixed_corrected,
                mixed_pending=mixed_pending,
                avg_sessions_to_complete=round(float(avg_sessions_to_complete), 2),
                completion_rate=round(completion_rate, 2),
                first_try_success_rate=round(first_try_success_rate, 2)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting lot summary: {e}")
            raise
    
    def _analyze_lot_success_patterns(self, db: Session) -> Dict[str, int]:
        """Analyze patterns of lot success/failure"""
        try:
            # Get all lots with their first session status
            lots_with_first_session = (
                db.query(
                    DetectionLot.id,
                    DetectionLot.is_target_match.label('lot_completed'),
                    func.min(DetectionSession.id).label('first_session_id')
                )
                .outerjoin(DetectionSession)
                .group_by(DetectionLot.id, DetectionLot.is_target_match)
                .subquery()
            )
            
            # Get first session results
            first_session_results = (
                db.query(
                    lots_with_first_session.c.id.label('lot_id'),
                    lots_with_first_session.c.lot_completed,
                    DetectionSession.is_target_match.label('first_session_success'),
                    func.count(DetectionSession.id).label('total_sessions')
                )
                .outerjoin(
                    DetectionSession,
                    DetectionSession.lot_id == lots_with_first_session.c.id
                )
                .group_by(
                    lots_with_first_session.c.id,
                    lots_with_first_session.c.lot_completed,
                    DetectionSession.is_target_match
                )
                .all()
            )
            
            # Categorize lots
            successful_first_try = 0
            mixed_corrected = 0
            mixed_pending = 0
            
            lot_status_map = {}
            for result in first_session_results:
                lot_id = result.lot_id
                if lot_id not in lot_status_map:
                    lot_status_map[lot_id] = {
                        'completed': result.lot_completed,
                        'first_success': result.first_session_success,
                        'total_sessions': result.total_sessions
                    }
            
            for lot_id, status in lot_status_map.items():
                if status['first_success'] and status['completed']:
                    successful_first_try += 1
                elif not status['first_success'] and status['completed']:
                    mixed_corrected += 1
                elif not status['completed'] and status['total_sessions'] > 0:
                    mixed_pending += 1
            
            return {
                'successful_first_try': successful_first_try,
                'mixed_corrected': mixed_corrected,
                'mixed_pending': mixed_pending
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing lot success patterns: {e}")
            return {'successful_first_try': 0, 'mixed_corrected': 0, 'mixed_pending': 0}
    
    def get_lots_by_time_period(self, db: Session, timeframe: str = 'monthly',
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> List[LotTimeStats]:
        """Get lot statistics grouped by time period"""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                if timeframe == 'daily':
                    start_date = end_date - timedelta(days=30)
                elif timeframe == 'weekly':
                    start_date = end_date - timedelta(weeks=12)
                elif timeframe == 'monthly':
                    start_date = end_date - timedelta(days=365)
                else:  # yearly
                    start_date = end_date - timedelta(days=365*3)

            # Build query based on timeframe
            if timeframe == 'daily':
                date_group = func.date(DetectionLot.created_at)
            elif timeframe == 'weekly':
                date_group = func.date_trunc('week', DetectionLot.created_at)
            elif timeframe == 'monthly':
                date_group = func.date_trunc('month', DetectionLot.created_at)
            else:  # yearly
                date_group = func.date_trunc('year', DetectionLot.created_at)

            # Query for lots created by period
            lots_created_query = (
                db.query(
                    date_group.label('period'),
                    func.count(DetectionLot.id).label('lots_created'),
                    func.sum(func.cast(DetectionLot.is_target_match, Integer)).label('lots_completed')
                )
                .filter(DetectionLot.created_at.between(start_date, end_date))
                .group_by(date_group)
                .order_by(date_group)
            )

            # For each period, analyze success patterns
            time_stats = []
            for row in lots_created_query.all():
                period_str = str(row.period)
                lots_created = row.lots_created
                lots_completed = row.lots_completed or 0
                
                # Get first-try success count for this period
                first_try_success_query = (
                    db.query(func.count(DetectionLot.id))
                    .join(DetectionSession)
                    .filter(
                        and_(
                            date_group == row.period,
                            DetectionSession.is_target_match == True
                        )
                    )
                    .group_by(DetectionLot.id)
                    .having(func.min(DetectionSession.created_at) == func.max(DetectionSession.created_at))
                )
                
                successful_first_try = first_try_success_query.scalar() or 0
                mixed_lots = lots_created - successful_first_try
                
                completion_rate = (lots_completed / lots_created * 100) if lots_created > 0 else 0
                first_try_success_rate = (successful_first_try / lots_created * 100) if lots_created > 0 else 0
                
                time_stats.append(LotTimeStats(
                    period=period_str,
                    lots_created=lots_created,
                    lots_completed=lots_completed,
                    successful_first_try=successful_first_try,
                    mixed_lots=mixed_lots,
                    completion_rate=round(completion_rate, 2),
                    first_try_success_rate=round(first_try_success_rate, 2)
                ))
            
            return time_stats
            
        except Exception as e:
            self.logger.error(f"Error getting lots by time period: {e}")
            raise

    def get_session_analytics(self, db: Session, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> SessionAnalytics:
        """Get comprehensive session analytics"""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)

            # Query session statistics
            session_stats_query = (
                db.query(
                    func.count(DetectionSession.id).label('total_sessions'),
                    func.sum(func.cast(DetectionSession.is_target_match, Integer)).label('successful_sessions'),
                    func.avg(DetectionSession.confidence_score).label('avg_confidence'),
                    func.avg(DetectionSession.total_pieces_detected).label('avg_pieces_detected'),
                    func.avg(DetectionSession.detection_rate).label('avg_detection_rate')
                )
                .filter(DetectionSession.created_at.between(start_date, end_date))
            )
            
            stats = session_stats_query.first()
            
            if not stats or stats.total_sessions == 0:
                return SessionAnalytics(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
            
            total_sessions = stats.total_sessions
            successful_sessions = stats.successful_sessions or 0
            failed_sessions = total_sessions - successful_sessions
            success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0
            
            return SessionAnalytics(
                total_sessions=total_sessions,
                successful_sessions=successful_sessions,
                failed_sessions=failed_sessions,
                avg_confidence=round(float(stats.avg_confidence or 0), 3),
                avg_pieces_detected=round(float(stats.avg_pieces_detected or 0), 2),
                avg_processing_time=0.0,  # Would need additional tracking
                success_rate=round(success_rate, 2)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting session analytics: {e}")
            raise

    def get_mix_up_analysis(self, db: Session, limit: int = 20) -> List[MixUpAnalysis]:
        """Analyze which piece groups get mixed up most often"""
        try:
            # Get lots that had mix-ups (first session failed but later succeeded or still pending)
            mixed_lots_query = (
                db.query(
                    DetectionLot.expected_piece_id,
                    func.count(DetectionLot.id).label('total_mixed_lots'),
                    func.avg(
                        db.query(func.count(DetectionSession.id))
                        .filter(DetectionSession.lot_id == DetectionLot.id)
                        .scalar_subquery()
                    ).label('avg_sessions_per_lot')
                )
                .join(DetectionSession)
                .filter(
                    DetectionSession.is_target_match == False  # First session was a failure
                )
                .group_by(DetectionLot.expected_piece_id)
                .order_by(func.count(DetectionLot.id).desc())
                .limit(limit)
            )

            mix_up_results = []
            for row in mixed_lots_query.all():
                expected_piece_id = row.expected_piece_id
                total_mixed_lots = row.total_mixed_lots
                avg_sessions = float(row.avg_sessions_per_lot or 0)

                # Get the most common wrong pieces detected for this expected piece
                wrong_pieces_query = (
                    db.query(
                        DetectedPiece.detected_label,
                        func.count(DetectedPiece.id).label('wrong_count')
                    )
                    .join(DetectionSession)
                    .join(DetectionLot)
                    .filter(
                        and_(
                            DetectionLot.expected_piece_id == expected_piece_id,
                            DetectedPiece.is_correct_piece == False,
                            DetectionSession.is_target_match == False
                        )
                    )
                    .group_by(DetectedPiece.detected_label)
                    .order_by(func.count(DetectedPiece.id).desc())
                    .limit(5)
                )

                wrong_pieces = [
                    {
                        'label': wp.detected_label,
                        'count': wp.wrong_count,
                        'percentage': round((wp.wrong_count / total_mixed_lots * 100), 2)
                    }
                    for wp in wrong_pieces_query.all()
                ]

                # Calculate mix-up rate (percentage of lots for this piece that get mixed up)
                total_lots_for_piece = (
                    db.query(func.count(DetectionLot.id))
                    .filter(DetectionLot.expected_piece_id == expected_piece_id)
                    .scalar() or 1
                )
                mix_up_rate = (total_mixed_lots / total_lots_for_piece * 100)

                # Get piece label (this would need a join to the Piece table if available)
                # For now, using piece_id as placeholder
                expected_piece_label = f"piece_{expected_piece_id}"

                mix_up_results.append(MixUpAnalysis(
                    expected_piece_id=expected_piece_id,
                    expected_piece_label=expected_piece_label,
                    total_mixed_lots=total_mixed_lots,
                    most_common_wrong_pieces=wrong_pieces,
                    avg_sessions_to_fix=round(avg_sessions, 2),
                    mix_up_rate=round(mix_up_rate, 2)
                ))

            return mix_up_results

        except Exception as e:
            self.logger.error(f"Error getting mix-up analysis: {e}")
            raise

    def get_lot_performance_by_piece(self, db: Session) -> List[Dict[str, Any]]:
        """Get performance statistics for each piece type in lots"""
        try:
            performance_query = (
                db.query(
                    DetectionLot.expected_piece_id,
                    func.count(DetectionLot.id).label('total_lots'),
                    func.sum(func.cast(DetectionLot.is_target_match, Integer)).label('successful_lots'),
                    func.avg(DetectionLot.expected_piece_number).label('avg_expected_pieces'),
                    func.avg(
                        db.query(func.count(DetectionSession.id))
                        .filter(DetectionSession.lot_id == DetectionLot.id)
                        .scalar_subquery()
                    ).label('avg_sessions_per_lot'),
                    func.avg(
                        db.query(func.avg(DetectionSession.confidence_score))
                        .filter(DetectionSession.lot_id == DetectionLot.id)
                        .scalar_subquery()
                    ).label('avg_confidence')
                )
                .group_by(DetectionLot.expected_piece_id)
                .order_by(func.count(DetectionLot.id).desc())
            )

            results = []
            for row in performance_query.all():
                total_lots = row.total_lots
                successful_lots = row.successful_lots or 0
                success_rate = (successful_lots / total_lots * 100) if total_lots > 0 else 0

                results.append({
                    'expected_piece_id': row.expected_piece_id,
                    'piece_label': f"piece_{row.expected_piece_id}",  # Would need actual label lookup
                    'total_lots': total_lots,
                    'successful_lots': successful_lots,
                    'failed_lots': total_lots - successful_lots,
                    'success_rate': round(success_rate, 2),
                    'avg_expected_pieces': round(float(row.avg_expected_pieces or 0), 1),
                    'avg_sessions_per_lot': round(float(row.avg_sessions_per_lot or 0), 2),
                    'avg_confidence': round(float(row.avg_confidence or 0), 3)
                })

            return results

        except Exception as e:
            self.logger.error(f"Error getting lot performance by piece: {e}")
            raise

    def get_session_failure_analysis(self, db: Session, days_back: int = 30) -> Dict[str, Any]:
        """Analyze reasons for session failures"""
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Get failed sessions with details
            failed_sessions_query = (
                db.query(
                    DetectionSession.id,
                    DetectionSession.lot_id,
                    DetectionLot.expected_piece_id,
                    DetectionLot.expected_piece_number,
                    DetectionSession.correct_pieces_count,
                    DetectionSession.misplaced_pieces_count,
                    DetectionSession.total_pieces_detected,
                    DetectionSession.confidence_score
                )
                .join(DetectionLot)
                .filter(
                    and_(
                        DetectionSession.created_at >= start_date,
                        DetectionSession.is_target_match == False
                    )
                )
            )

            failure_reasons = {
                'wrong_count': 0,
                'missing_pieces': 0,
                'extra_pieces': 0,
                'low_confidence': 0,
                'wrong_pieces_detected': 0,
                'total_failed_sessions': 0
            }

            for session in failed_sessions_query.all():
                failure_reasons['total_failed_sessions'] += 1
                
                expected_count = session.expected_piece_number
                detected_count = session.total_pieces_detected
                correct_count = session.correct_pieces_count
                confidence = session.confidence_score or 0
                
                # Categorize failure reasons
                if detected_count < expected_count:
                    failure_reasons['missing_pieces'] += 1
                elif detected_count > expected_count:
                    failure_reasons['extra_pieces'] += 1
                elif correct_count != expected_count:
                    failure_reasons['wrong_count'] += 1
                
                if confidence < 0.5:  # Low confidence threshold
                    failure_reasons['low_confidence'] += 1
                
                if session.misplaced_pieces_count > 0:
                    failure_reasons['wrong_pieces_detected'] += 1

            # Calculate percentages
            total_failed = failure_reasons['total_failed_sessions']
            if total_failed > 0:
                for key in failure_reasons:
                    if key != 'total_failed_sessions':
                        percentage = (failure_reasons[key] / total_failed * 100)
                        failure_reasons[f'{key}_percentage'] = round(percentage, 2)

            return failure_reasons

        except Exception as e:
            self.logger.error(f"Error getting session failure analysis: {e}")
            raise

    def get_lot_completion_trends(self, db: Session, days_back: int = 90) -> Dict[str, Any]:
        """Get trends in lot completion over time"""
        try:
            start_date = datetime.now() - timedelta(days=days_back)
            
            # Get daily completion trends
            daily_trends_query = (
                db.query(
                    func.date(DetectionLot.created_at).label('date'),
                    func.count(DetectionLot.id).label('lots_created'),
                    func.sum(func.cast(DetectionLot.is_target_match, Integer)).label('lots_completed')
                )
                .filter(DetectionLot.created_at >= start_date)
                .group_by(func.date(DetectionLot.created_at))
                .order_by(func.date(DetectionLot.created_at))
            )

            trends = []
            for row in daily_trends_query.all():
                completion_rate = (row.lots_completed / row.lots_created * 100) if row.lots_created > 0 else 0
                trends.append({
                    'date': str(row.date),
                    'lots_created': row.lots_created,
                    'lots_completed': row.lots_completed or 0,
                    'completion_rate': round(completion_rate, 2)
                })

            # Calculate overall trend metrics
            if len(trends) >= 2:
                recent_avg = sum(t['completion_rate'] for t in trends[-7:]) / min(7, len(trends))
                older_avg = sum(t['completion_rate'] for t in trends[:7]) / min(7, len(trends))
                trend_direction = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
            else:
                recent_avg = trends[0]['completion_rate'] if trends else 0
                trend_direction = "stable"

            return {
                'daily_trends': trends,
                'trend_analysis': {
                    'recent_avg_completion_rate': round(recent_avg, 2),
                    'trend_direction': trend_direction,
                    'total_days_analyzed': len(trends)
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting lot completion trends: {e}")
            raise

    def export_lot_statistics_report(self, db: Session, 
                                   timeframe: str = 'monthly',
                                   days_back: int = 30) -> Dict[str, Any]:
        """Export comprehensive lot statistics report"""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'timeframe': timeframe,
                'analysis_period_days': days_back,
                'summary': self.get_overall_lot_summary(db).__dict__,
                'time_series': [stat.__dict__ for stat in self.get_lots_by_time_period(db, timeframe)],
                'session_analytics': self.get_session_analytics(db).__dict__,
                'mix_up_analysis': [analysis.__dict__ for analysis in self.get_mix_up_analysis(db)],
                'piece_performance': self.get_lot_performance_by_piece(db),
                'failure_analysis': self.get_session_failure_analysis(db, days_back),
                'completion_trends': self.get_lot_completion_trends(db, days_back)
            }
            
            self.logger.info(f"Generated comprehensive lot statistics report with {len(report['time_series'])} time periods")
            return report
            
        except Exception as e:
            self.logger.error(f"Error exporting lot statistics report: {e}")
            raise

    def get_lots_needing_attention(self, db: Session, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """Get lots that need attention (pending, multiple failed sessions, etc.)"""
        try:
            # Lots with no sessions
            lots_no_sessions = (
                db.query(DetectionLot)
                .outerjoin(DetectionSession)
                .filter(DetectionSession.id.is_(None))
                .order_by(DetectionLot.created_at.asc())
                .limit(limit)
                .all()
            )

            # Lots with multiple failed sessions
            lots_multiple_failures_query = (
                db.query(
                    DetectionLot.id,
                    DetectionLot.lot_name,
                    DetectionLot.expected_piece_id,
                    DetectionLot.expected_piece_number,
                    DetectionLot.created_at,
                    func.count(DetectionSession.id).label('failed_sessions')
                )
                .join(DetectionSession)
                .filter(
                    and_(
                        DetectionLot.is_target_match == False,
                        DetectionSession.is_target_match == False
                    )
                )
                .group_by(
                    DetectionLot.id,
                    DetectionLot.lot_name,
                    DetectionLot.expected_piece_id,
                    DetectionLot.expected_piece_number,
                    DetectionLot.created_at
                )
                .having(func.count(DetectionSession.id) >= 3)  # 3+ failed attempts
                .order_by(func.count(DetectionSession.id).desc())
                .limit(limit)
            )

            # Old pending lots (created more than 24 hours ago but not completed)
            old_pending_lots = (
                db.query(DetectionLot)
                .filter(
                    and_(
                        DetectionLot.is_target_match == False,
                        DetectionLot.created_at < datetime.now() - timedelta(hours=24)
                    )
                )
                .order_by(DetectionLot.created_at.asc())
                .limit(limit)
                .all()
            )

            return {
                'no_sessions': [
                    {
                        'id': lot.id,
                        'lot_name': lot.lot_name,
                        'expected_piece_id': lot.expected_piece_id,
                        'expected_piece_number': lot.expected_piece_number,
                        'created_at': lot.created_at.isoformat() if lot.created_at else None,
                        'age_hours': (datetime.now() - lot.created_at).total_seconds() / 3600 if lot.created_at else 0
                    }
                    for lot in lots_no_sessions
                ],
                'multiple_failures': [
                    {
                        'id': lot.id,
                        'lot_name': lot.lot_name,
                        'expected_piece_id': lot.expected_piece_id,
                        'expected_piece_number': lot.expected_piece_number,
                        'created_at': lot.created_at.isoformat() if lot.created_at else None,
                        'failed_sessions': lot.failed_sessions
                    }
                    for lot in lots_multiple_failures_query.all()
                ],
                'old_pending': [
                    {
                        'id': lot.id,
                        'lot_name': lot.lot_name,
                        'expected_piece_id': lot.expected_piece_id,
                        'expected_piece_number': lot.expected_piece_number,
                        'created_at': lot.created_at.isoformat() if lot.created_at else None,
                        'age_hours': (datetime.now() - lot.created_at).total_seconds() / 3600 if lot.created_at else 0
                    }
                    for lot in old_pending_lots
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting lots needing attention: {e}")
            raise

# Global lot statistics service
lot_statistics_service = LotStatisticsService()