from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware




def create_application():
    application = FastAPI(
        title="Artifact Keeper Service",
        description="Microservice for managing artifacts and cameras",
        version="1.0.0"
    )
   
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
