import os
import time

import jwt
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter()

MOCK_USERS = {
    "eun": "seo",
}

# 서비스 로그인 세션 유효시간 — 시연(최대 3시간) 중 만료되지 않도록 180분으로 둔다.
# Tableau 임베딩 JWT(tableau.py, 최대 10분)와는 별개이며, 그쪽은 프론트가 자동 갱신한다.
SESSION_TTL_SECONDS = 3 * 60 * 60

# 서명 키는 src/backend/.env 의 AUTH_SECRET_KEY 로 교체한다. 기본값은 로컬 시연용.
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "capstone-local-demo-secret")
AUTH_ALGORITHM = "HS256"


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type:   str
    username:     str
    expires_in:   int


def decode_access_token(token: str) -> str:
    """서비스 로그인 토큰을 검증하고 username(sub)을 반환한다. 만료/위조 시 401."""
    try:
        payload = jwt.decode(token, AUTH_SECRET_KEY, algorithms=[AUTH_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인 세션이 만료되었습니다. 다시 로그인해 주세요.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 로그인 토큰입니다. 다시 로그인해 주세요.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload["sub"]


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    if MOCK_USERS.get(body.username) != body.password:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    now = int(time.time())
    token = jwt.encode(
        {"sub": body.username, "iat": now, "exp": now + SESSION_TTL_SECONDS},
        AUTH_SECRET_KEY,
        algorithm=AUTH_ALGORITHM,
    )
    return {
        "access_token": token,
        "token_type":   "bearer",
        "username":     body.username,
        "expires_in":   SESSION_TTL_SECONDS,
    }
