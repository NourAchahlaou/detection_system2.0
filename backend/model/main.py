from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


import uvicorn

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


# uvicorn main:app --host 127.0.0.1 --port 8001 --reload
