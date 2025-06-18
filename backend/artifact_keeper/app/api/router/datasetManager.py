from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Annotated


from artifact_keeper.app.services.datasetManagerService import get_all_datasets,delete_all_pieces,delete_piece_by_label
from artifact_keeper.app.db.session import get_session

datasetManager_router = APIRouter(
    prefix="/datasetManager",
    tags=["dgataset Manager"],
    responses={404: {"description": "Not found"}},
)
db_dependency = Annotated[Session, Depends(get_session)]

@datasetManager_router.get("/datasets", tags=["Dataset"])
def get_datasets_route(db: Session = Depends(get_session)):
    """Route to fetch all datasets."""
    datasets = get_all_datasets(db)
    if not datasets:
        raise HTTPException(status_code=404, detail="No datasets found")
    return datasets



@datasetManager_router.delete("/delete_piece/{piece_label}")
def delete_piece(piece_label: str, db: db_dependency):
    return delete_piece_by_label(piece_label, db)

@datasetManager_router.delete("/delete_all_pieces")
def delete_all_pieces_route(db: db_dependency):
    return delete_all_pieces(db)