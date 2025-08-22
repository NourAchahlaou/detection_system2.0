import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.route import camera, camera_hardware_profiling_router


def create_application():
    application = FastAPI()
    application.include_router(camera.camera_router)
    application.include_router(camera_hardware_profiling_router.profiling_router)
    return application



app = create_application()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "this is the piece registry service"}


# uvicorn main:app --host 127.0.0.1 --port 8003  
