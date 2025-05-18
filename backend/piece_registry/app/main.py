from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from piece_registry.app.db.session import get_session
from piece_registry.app.service.camera_manager import CameraManager
from piece_registry.app.service.external_camera import get_available_cameras
from piece_registry.app.api.route import camera

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
    db = next(get_session())

    CameraManager.detect_and_save_cameras(db)
   
    cameras = get_available_cameras()
    print("Available Cameras:")
    for camera in cameras:
        print(camera)


@app.get("/")
async def root():
    return {"message": "this is the piece registry service"}

