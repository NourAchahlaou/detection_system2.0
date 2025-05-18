
# Import all models to register them with Base.metadata
# NOTE: These imports are for registering models with SQLAlchemy
# Don't use these imports directly in your application code
from piece_registry.app.db.session import Base
from piece_registry.app.db.models.camera import Camera
from piece_registry.app.db.models.piece import Piece
from piece_registry.app.db.models.camera_settings import CameraSettings
from piece_registry.app.db.models.piece_docs import PieceDocument
from piece_registry.app.db.models.piece_image import PieceImage  

# Define __all__ to control what gets imported with "from models import *"
__all__ = ['Camera', 'Piece', 'Shift', 'CameraSettings', 'PieceDocument', 'PieceImage']
