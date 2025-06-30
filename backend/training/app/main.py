from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_application():
    application = FastAPI(
        title="dataset Service",
        description="Microservice for managing database operations in the dataset application",
        version="1.0.0"
    )
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
        "message": " training Service", 
        "status": "running",
        "service": "training",
    }

