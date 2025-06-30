from pydantic import BaseModel


class TrainRequest(BaseModel):
    piece_labels: list[str]