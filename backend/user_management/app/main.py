from fastapi import FastAPI
from user_management.app.api.routes import user, profile,profileRouter,accountRouter,shiftaccountRouter
from fastapi.middleware.cors import CORSMiddleware

def create_application():
    application = FastAPI()
    application.include_router(user.user_router)
    application.include_router(user.guest_router)
    application.include_router(user.auth_router)
    application.include_router(profile.profile_router)
    application.include_router(profile.guest_router)
    application.include_router(profileRouter.profile_tab_router)   
    application.include_router(accountRouter.profile_router)
    application.include_router(shiftaccountRouter.shift_router)
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
    return {"message": "this is the user management service"}

