import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.service.camera_manager import CameraManager
from app.service.external_camera import get_available_cameras
from app.db.session import get_session
from app.api.route import camera

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

if __name__ == "__main__":
    import uvicorn
    if os.getenv('ENVIRONMENT') == 'development':
        uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
    else :
        uvicorn.run(app, host="127.0.0.1", port=8001)
