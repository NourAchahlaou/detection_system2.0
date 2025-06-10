# Add this to your artifact_keeper FastAPI router

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Add this endpoint to serve images from the dataset
@router.get("/images/{file_path:path}")
async def serve_image(file_path: str):
    """Serve images from the dataset directory."""
    try:
        # Base dataset path from your docker compose
        dataset_base_path = "/app/shared/dataset"
        
        # Construct full file path
        full_path = os.path.join(dataset_base_path, file_path)
        
        # Security check: make sure the path is within the dataset directory
        real_dataset_path = os.path.realpath(dataset_base_path)
        real_file_path = os.path.realpath(full_path)
        
        if not real_file_path.startswith(real_dataset_path):
            logger.warning(f"Attempted path traversal attack: {file_path}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not os.path.exists(full_path):
            logger.warning(f"Image file not found: {full_path}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Check if it's actually an image file
        if not full_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            logger.warning(f"Invalid image file requested: {full_path}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Serving image: {full_path}")
        return FileResponse(
            path=full_path,
            media_type="image/jpeg",  # Adjust based on your image types
            filename=os.path.basename(full_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to serve image")