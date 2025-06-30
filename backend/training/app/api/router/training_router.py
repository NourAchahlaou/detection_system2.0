import logging
import threading
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from training.app.request.training_request import TrainRequest
from training.app.services.model_training_service import train_model, stop_training
from training.app.db.session import get_session



# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("training_logs.log", mode='a')])

logger = logging.getLogger(__name__)

training_router = APIRouter(
    prefix="/training",
    tags=["Training"],
    responses={404: {"description": "Not found"}},
)
db_dependency = Annotated[Session, Depends(get_session)]
stop_event = threading.Event()  # Event to signal when to stop

@training_router.post("/train")
def train_piece_model(request: TrainRequest, db: db_dependency):
    try:
        train_model(request.piece_labels, db)
        return {"message": "Training process started. Check logs for updates."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
@training_router.post("/stop_training")
async def stop_training_yolo():
    try:
        await stop_training()
        return {"message": "Stop training signal sent."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")