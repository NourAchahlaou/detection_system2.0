from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from detection.app.api.route.detection_redis_router import redis_router
from detection.app.api.route.graceful_shutdown_endpoints import detection_shutdown_router   
from detection.app.api.route.basic_detection_router import basic_detection_router
from detection.app.api.route.statistics_service_router import statistic_detection_router 
from detection.app.api.route.identification_router import identification_router
from detection.app.api.route.lot_session_router import lot_dashboard_router
def create_application():
    application = FastAPI(
        title="detection Service",
        description="Microservice for managing detections",
        version="1.0.0"
    )
   
    application.include_router(redis_router)
    application.include_router(detection_shutdown_router)
    application.include_router(basic_detection_router)
    application.include_router(statistic_detection_router)
    application.include_router(identification_router)
    application.include_router(lot_dashboard_router)
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
