from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
from artifact_keeper.app.db.models.piece import Piece


def get_piece_statistics(db: Session):
    now = datetime.utcnow()

    # Time ranges
    start_of_day = datetime(now.year, now.month, now.day)
    start_of_week = now - timedelta(days=7)
    start_of_month = now - timedelta(days=30)

    stats = {
        "created_today": db.query(func.count(Piece.id)).filter(Piece.created_at >= start_of_day).scalar(),
        "created_this_week": db.query(func.count(Piece.id)).filter(Piece.created_at >= start_of_week).scalar(),
        "created_this_month": db.query(func.count(Piece.id)).filter(Piece.created_at >= start_of_month).scalar(),
        "total": db.query(func.count(Piece.id)).scalar(),
        "annotated": db.query(func.count(Piece.id)).filter(Piece.is_annotated == True).scalar(),
        "trained": db.query(func.count(Piece.id)).filter(Piece.is_yolo_trained == True).scalar(),
    }

    return stats
