from pydantic import BaseModel

class CleanupResponse(BaseModel):
    message: str

