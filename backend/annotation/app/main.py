from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from annotation.app.db.session import get_session
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
    return {"message": "this is the artifact keeper service"}

