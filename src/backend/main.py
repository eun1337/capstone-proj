import os
from pathlib import Path

# src/backend/.env 파일의 환경변수를 자동으로 로드한다.
# python-dotenv가 없으면 OS 환경변수만 사용하므로 동작에는 문제없다.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass  # python-dotenv 미설치 시 OS 환경변수만 사용

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from routers import auth, tableau, logistics, dashboard, model_analysis

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
app.include_router(model_analysis.router, prefix="/api/model-analysis", tags=["model-analysis"])


@app.get("/")
def root():
    return {"message": "물류 수요 예측 API 정상 동작 중"}
