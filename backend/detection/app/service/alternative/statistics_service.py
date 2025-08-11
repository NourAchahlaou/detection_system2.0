# statistics_service.py - Professional Detection Statistics Service
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import func, desc, and_, literal_column
from datetime import datetime

from detection.app.db.models.detectionLot import DetectionLot
from detection.app.db.models.detectionSession import DetectionSession

logger = logging.getLogger(__name__)


# --- Data containers (clear, typed outputs) ---
@dataclass
class LotLastSession:
    lot_id: int
    lot_name: str
    last_session_id: int
    last_session_time: str
    correct_pieces: int
    misplaced_pieces: int
    total_detected: int
    is_target_match: bool
    confidence_score: float
    detection_rate: float


@dataclass
class LotSummary:
    lot_id: int
    lot_name: str
    expected_piece_id: int
    expected_piece_number: int
    created_at: str
    completed_at: Optional[str]
    is_completed: bool
    sessions_count: int
    sessions_to_completion: Optional[int]  # None if not completed
    first_session_correct: bool
    total_correct: int
    total_misplaced: int


@dataclass
class SystemLotStartStats:
    total_lots: int
    lots_correct_from_start: int
    lots_with_problems_from_start: int
    percent_correct_from_start: float
    percent_problem_from_start: float


@dataclass
class CommonFailure:
    description: str
    count: int
    percent_of_problem_lots: float


@dataclass
class MixedPair:
    piece_a_id: int
    piece_b_id: int
    confusion_count: int


# --- Core service ---
class DetectionStatisticsService:
    """
    Focused, professional statistics service.
    Methods:
      - last_session_per_lot(db)
      - lot_summary(lot_id, db)
      - sessions_to_completion(lot_id, db)
      - system_start_stats(db)
      - common_failures_for_problem_lots(db, top_n=10)
      - top_mixed_pairs(db, top_n=10)  # needs mismatch details or falls back to heuristics
    """

    def __init__(self, cache_timeout: int = 30):
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, float] = {}
        self.cache_timeout = cache_timeout

    def _cache_get(self, key: str):
        ts = self._cache_ts.get(key)
        if ts and (time.time() - ts) < self.cache_timeout:
            return self._cache.get(key)
        return None

    def _cache_set(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    # -------- Last session for each lot --------
    def last_session_per_lot(self, db: Session) -> List[LotLastSession]:
        """
        Returns last session (most recent) for every lot with brief metrics.
        Efficient: single query to get last session per lot via subquery.
        """
        cache_key = "last_session_per_lot"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # Subquery to get max(created_at) per lot
            subq = db.query(
                DetectionSession.lot_id.label("lot_id"),
                func.max(DetectionSession.created_at).label("last_time")
            ).group_by(DetectionSession.lot_id).subquery()

            # Join subquery with sessions and lots to fetch session row details
            rows = db.query(
                DetectionLot.id.label("lot_id"),
                DetectionLot.lot_name,
                DetectionSession.id.label("session_id"),
                DetectionSession.created_at,
                DetectionSession.correct_pieces_count,
                DetectionSession.misplaced_pieces_count,
                DetectionSession.total_pieces_detected,
                DetectionSession.is_target_match,
                DetectionSession.confidence_score,
                DetectionSession.detection_rate
            ).join(DetectionSession, DetectionSession.lot_id == DetectionLot.id) \
             .join(subq, and_(subq.c.lot_id == DetectionSession.lot_id,
                               subq.c.last_time == DetectionSession.created_at)) \
             .all()

            result = []
            for r in rows:
                result.append(LotLastSession(
                    lot_id=r.lot_id,
                    lot_name=r.lot_name,
                    last_session_id=r.session_id,
                    last_session_time=r.created_at.isoformat() if r.created_at else datetime.utcnow().isoformat(),
                    correct_pieces=int(r.correct_pieces_count or 0),
                    misplaced_pieces=int(r.misplaced_pieces_count or 0),
                    total_detected=int(r.total_pieces_detected or ( (r.correct_pieces_count or 0) + (r.misplaced_pieces_count or 0) )),
                    is_target_match=bool(r.is_target_match),
                    confidence_score=float(r.confidence_score or 0.0),
                    detection_rate=float(r.detection_rate or 0.0)
                ))
            self._cache_set(cache_key, result)
            return result

        except Exception as e:
            logger.exception("Error computing last_session_per_lot: %s", e)
            return []

    # -------- Lot summary & sessions-to-completion --------
    def lot_summary(self, lot_id: int, db: Session) -> Optional[LotSummary]:
        """
        Returns an aggregated summary for the given lot:
         - session counts
         - total correct / misplaced
         - whether first session was correct (i.e. no misplaced and correct==expected)
         - sessions to completion (index of session that achieved completion), if completed
        """
        cache_key = f"lot_summary_{lot_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            lot: DetectionLot = db.query(DetectionLot).options(selectinload(DetectionLot.detection_sessions)) \
                .filter(DetectionLot.id == lot_id).first()
            if not lot:
                return None

            sessions = sorted(lot.detection_sessions, key=lambda s: s.created_at or datetime.min)
            sessions_count = len(sessions)
            total_correct = sum(int(s.correct_pieces_count or 0) for s in sessions)
            total_misplaced = sum(int(s.misplaced_pieces_count or 0) for s in sessions)
            expected = int(lot.expected_piece_number or 0)

            # First session correctness: no misplaced AND correct == expected (strict)
            first_session_correct = False
            if sessions_count > 0:
                first = sessions[0]
                first_correct = int(first.correct_pieces_count or 0)
                first_mis = int(first.misplaced_pieces_count or 0)
                first_session_correct = (first_mis == 0) and (expected > 0 and first_correct >= expected)

            # Sessions to completion: find first session index where cumulative correct >= expected and misplaced == 0 (or is_target_match True)
            sessions_to_completion = None
            if expected > 0:
                cumulative_correct = 0
                for idx, s in enumerate(sessions, start=1):
                    cumulative_correct += int(s.correct_pieces_count or 0)
                    # prefer the lot/session's own is_target_match flag if present
                    session_completed_flag = getattr(s, "is_target_match", False)
                    if session_completed_flag or (cumulative_correct >= expected and int(s.misplaced_pieces_count or 0) == 0):
                        sessions_to_completion = idx
                        break

            summary = LotSummary(
                lot_id=lot.id,
                lot_name=lot.lot_name,
                expected_piece_id=int(lot.expected_piece_id or 0),
                expected_piece_number=int(lot.expected_piece_number or 0),
                created_at=lot.created_at.isoformat() if lot.created_at else datetime.utcnow().isoformat(),
                completed_at=lot.completed_at.isoformat() if getattr(lot, "completed_at", None) else None,
                is_completed=bool(getattr(lot, "is_target_match", False)),
                sessions_count=sessions_count,
                sessions_to_completion=sessions_to_completion,
                first_session_correct=first_session_correct,
                total_correct=total_correct,
                total_misplaced=total_misplaced
            )
            self._cache_set(cache_key, summary)
            return summary

        except Exception as e:
            logger.exception("Error computing lot_summary for %s: %s", lot_id, e)
            return None

    def sessions_to_completion(self, lot_id: int, db: Session) -> Optional[int]:
        """Convenience: returns sessions_to_completion for lot (None if not completed)"""
        summary = self.lot_summary(lot_id, db)
        return summary.sessions_to_completion if summary else None

    # -------- System-level start-state statistics --------
    def system_start_stats(self, db: Session) -> SystemLotStartStats:
        """
        Tells how many lots were correct from the first session vs had problems from the first session.
        A lot is considered 'correct from start' if first session has 0 misplaced and correct >= expected.
        """
        cache_key = "system_start_stats"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            lots = db.query(DetectionLot).options(selectinload(DetectionLot.detection_sessions)).all()
            total = len(lots)
            correct_from_start = 0
            problem_from_start = 0

            for lot in lots:
                sessions = sorted(lot.detection_sessions, key=lambda s: s.created_at or datetime.min)
                if not sessions:
                    # No sessions - treat as problem_from_start (or ignore). We'll treat as problem.
                    problem_from_start += 1
                    continue
                first = sessions[0]
                expected = int(lot.expected_piece_number or 0)
                first_correct = int(getattr(first, "correct_pieces_count", 0) or 0)
                first_mis = int(getattr(first, "misplaced_pieces_count", 0) or 0)
                if expected > 0 and first_mis == 0 and first_correct >= expected:
                    correct_from_start += 1
                else:
                    problem_from_start += 1

            percent_ok = (correct_from_start / total * 100) if total > 0 else 0.0
            percent_problem = (problem_from_start / total * 100) if total > 0 else 0.0

            stats = SystemLotStartStats(
                total_lots=total,
                lots_correct_from_start=correct_from_start,
                lots_with_problems_from_start=problem_from_start,
                percent_correct_from_start=round(percent_ok, 1),
                percent_problem_from_start=round(percent_problem, 1)
            )
            self._cache_set(cache_key, stats)
            return stats

        except Exception as e:
            logger.exception("Error computing system_start_stats: %s", e)
            return SystemLotStartStats(0, 0, 0, 0.0, 0.0)

    # -------- Common failure analysis for problem lots --------
    def common_failures_for_problem_lots(self, db: Session, top_n: int = 10) -> List[CommonFailure]:
        """
        Returns top common problems observed in lots that had problems from first session.
        Problems are heuristically categorized:
          - 'mismatched_count' : many misplaced pieces in first session
          - 'missing_pieces'   : first session correct < expected
          - 'other'            : anything else (low confidence, high detection_rate drop, etc.)
        If you have a structured 'misplaced_details' JSON column in DetectionSession the service can be extended to extract types.
        """
        cache_key = f"common_failures_{top_n}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            problem_counts = {"mismatched_count": 0, "missing_pieces": 0, "no_sessions": 0, "other": 0}
            total_problem_lots = 0

            lots = db.query(DetectionLot).options(selectinload(DetectionLot.detection_sessions)).all()
            for lot in lots:
                sessions = sorted(lot.detection_sessions, key=lambda s: s.created_at or datetime.min)
                if not sessions:
                    total_problem_lots += 1
                    problem_counts["no_sessions"] += 1
                    continue
                first = sessions[0]
                expected = int(lot.expected_piece_number or 0)
                first_correct = int(getattr(first, "correct_pieces_count", 0) or 0)
                first_mis = int(getattr(first, "misplaced_pieces_count", 0) or 0)

                # classify
                if expected > 0 and first_mis > 0:
                    problem_counts["mismatched_count"] += 1
                    total_problem_lots += 1
                elif expected > 0 and first_correct < expected:
                    problem_counts["missing_pieces"] += 1
                    total_problem_lots += 1
                elif first_mis == 0 and first_correct >= expected:
                    # correct_from_start -> not counted
                    pass
                else:
                    problem_counts["other"] += 1
                    total_problem_lots += 1

            failures: List[CommonFailure] = []
            if total_problem_lots == 0:
                return failures

            for k, v in problem_counts.items():
                if v == 0:
                    continue
                pct = (v / total_problem_lots * 100) if total_problem_lots > 0 else 0.0
                desc = {
                    "mismatched_count": "Misplaced / mismatched pieces in first session",
                    "missing_pieces": "First session had fewer correct pieces than expected (missing)",
                    "no_sessions": "No sessions recorded for lot",
                    "other": "Other problems (low confidence, partial matches, etc.)"
                }.get(k, k)
                failures.append(CommonFailure(description=desc, count=v, percent_of_problem_lots=round(pct, 1)))

            # sort by count
            failures.sort(key=lambda x: x.count, reverse=True)
            failures = failures[:top_n]
            self._cache_set(cache_key, failures)
            return failures

        except Exception as e:
            logger.exception("Error computing common_failures_for_problem_lots: %s", e)
            return []

    # -------- Most mixed-up pieces (confusion pairs) --------
    def top_mixed_pairs(self, db: Session, top_n: int = 10) -> List[MixedPair]:
        """
        Returns most confused piece pairs.
        Preferred approach: DB contains per-session JSON column 'misplaced_details' with list of dicts like:
          [{"expected": <id>, "detected_as": <id>, "count": n}, ...]
        If that column exists, we aggregate it. Otherwise, we derive a heuristic:
          - For each lot (expected_piece_id), sum misplaced counts across sessions and attribute them to 'unknown' detected ids not available.
        NOTE: This function is defensive — it will gracefully fall back if structured data is absent.
        """
        cache_key = f"top_mixed_pairs_{top_n}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # First try: structured JSON column named 'misplaced_details' on DetectionSession
            sample_session = db.query(DetectionSession).limit(1).first()
            if sample_session and hasattr(sample_session, "misplaced_details") and sample_session.misplaced_details:
                # We assume misplaced_details is a JSON blob stored as a Python structure by the ORM
                pair_counts: Dict[Tuple[int, int], int] = {}
                sessions = db.query(DetectionSession).all()
                for s in sessions:
                    details = getattr(s, "misplaced_details", None)
                    if not details:
                        continue
                    # details expected to be iterable of dicts: {"expected": id, "detected_as": id, "count": n}
                    try:
                        for d in details:
                            a = int(d.get("expected") or 0)
                            b = int(d.get("detected_as") or 0)
                            c = int(d.get("count") or 1)
                            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + c
                    except Exception:
                        # ignore malformed entries
                        continue

                pairs = [MixedPair(piece_a_id=a, piece_b_id=b, confusion_count=c) for (a, b), c in pair_counts.items()]
                pairs.sort(key=lambda p: p.confusion_count, reverse=True)
                pairs = pairs[:top_n]
                self._cache_set(cache_key, pairs)
                return pairs

            # Fallback: aggregate misplaced counts by expected_piece_id (we cannot know detected id),
            # Return top expected pieces that were most misplaced (as pair with 0 detected id)
            agg = db.query(
                DetectionLot.expected_piece_id.label("expected_id"),
                func.sum(DetectionSession.misplaced_pieces_count).label("misplaced_sum")
            ).join(DetectionSession, DetectionSession.lot_id == DetectionLot.id) \
             .group_by(DetectionLot.expected_piece_id).order_by(desc("misplaced_sum")).limit(top_n).all()

            result = []
            for row in agg:
                result.append(MixedPair(piece_a_id=int(row.expected_id or 0), piece_b_id=0, confusion_count=int(row.misplaced_sum or 0)))

            self._cache_set(cache_key, result)
            return result

        except Exception as e:
            logger.exception("Error computing top_mixed_pairs: %s", e)
            return []

    # -------- Utility: clear cache --------
    def clear_cache(self):
        self._cache.clear()
        self._cache_ts.clear()
        logger.info("Statistics cache cleared.")


# Single instance (fixed: before was a tuple in your file)
detection_statistics_service = DetectionStatisticsService()
