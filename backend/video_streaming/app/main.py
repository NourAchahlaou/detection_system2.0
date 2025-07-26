from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from video_streaming.app.api.route.videoStreamRouter import video_router
from video_streaming.app.api.route.videoStream_detection import detection_streaming_router
from video_streaming.app.api.route.video_redis_router import optimized_router

def create_application():
    application = FastAPI(
        title="video streaming Service",
        description="Microservice for managing video streaming",
        version="1.0.0"
    )
    application.include_router(video_router)
    application.include_router(detection_streaming_router)
    application.include_router(optimized_router)
    return application

app = create_application()
    # Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
    
# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "detection Service", 
        "status": "running",
        "service": "detection"
    }
