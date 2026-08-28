from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, tableau, logistics, dashboard

app = FastAPI(title="물류 수요 예측 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/api", tags=["auth"])
app.include_router(tableau.router,   prefix="/api", tags=["tableau"])
app.include_router(logistics.router, prefix="/api", tags=["logistics"])
app.include_router(dashboard.router,  prefix="/api/dashboard", tags=["dashboard"])


@app.get("/")
def root():
    return {"message": "물류 수요 예측 API 정상 동작 중"}
