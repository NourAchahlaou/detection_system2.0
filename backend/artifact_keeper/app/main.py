from fastapi import FastAPI
from artifact_keeper.app.api.router import camera
from fastapi.middleware.cors import CORSMiddleware

def create_application():
    application = FastAPI()
    application.include_router(camera.camera_router)

    return application





app = create_application()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    from artifact_keeper.app.services.camera import CameraService
    # Initialize camera manager and detect cameras
    CameraService.test_camera_detection()

@app.get("/")
async def root():
    return {"message": "this is the artifact keeper service"}

