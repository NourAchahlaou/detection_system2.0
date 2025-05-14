from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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



@app.get("/")
async def root():
    return {"message": "this is the piece registry service"}

