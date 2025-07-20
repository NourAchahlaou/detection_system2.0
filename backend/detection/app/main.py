from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from detection.app.api.route.detection_route import detection_router



def create_application():
    application = FastAPI(
        title="detection Service",
        description="Microservice for managing detections",
        version="1.0.0"
    )
    application.include_router(detection_router)
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
