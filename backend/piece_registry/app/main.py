from fastapi import FastAPI
from user_management.app.api.routes import user, profile
from fastapi.middleware.cors import CORSMiddleware

def create_application():
    application = FastAPI()
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

