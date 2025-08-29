from fastapi import FastAPI
from artifact_keeper.app.api.router import camera, health, dataset, captureImage,datasetManager,system_spec,pieceStatisticsRouter
from fastapi.middleware.cors import CORSMiddleware
from artifact_keeper.app.db.session import get_session

def create_application():
    application = FastAPI(
        title="Artifact Keeper Service",
        description="Microservice for managing artifacts and cameras",
        version="1.0.0"
    )
    application.include_router(camera.camera_router)
    application.include_router(health.health_router)
    application.include_router(dataset.router)
    application.include_router(captureImage.captureImage_router)
    application.include_router(datasetManager.datasetManager_router)    
    application.include_router(system_spec.system_router)
    application.include_router(pieceStatisticsRouter.router)
    return application

app = create_application()

# Fixed CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost",       # Nginx gateway
        "http://172.23.0.5:3000", # Docker network access
        "*"  # For development - restrict in production
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Artifact Keeper Service", 
        "status": "running",
        "service": "artifact_keeper"
    }

# Simple health check at root level (for Docker health checks)
@app.get("/health")
async def simple_health():
    return {"status": "healthy", "service": "artifact_keeper"}

@app.on_event("startup")
async def startup_event():
    from artifact_keeper.app.services.camera import CameraService
    db = next(get_session())
    cameraservice= CameraService()
    # Initialize camera manager and detect cameras
    cameraservice.detect_and_save_cameras(db)