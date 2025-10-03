from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from artifact_keeper.app.db.session import get_session
from artifact_keeper.app.services.statistic.statistics_PieceService import piece_statistics_service

router = APIRouter(
    prefix="/statistics",
    tags=["Statistics"]
)

@router.get("/pieces")
def piece_statistics(db: Session = Depends(get_session)):
    """
    Returns statistics about pieces:
    - created today / this week / this month
    - total created
    - annotated pieces
    - trained pieces
    """
    return piece_statistics_service(db)
