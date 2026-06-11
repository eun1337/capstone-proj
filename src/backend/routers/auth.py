from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

MOCK_USERS = {
    "admin": "password123",
    "user": "user123",
}


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    username: str


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    if MOCK_USERS.get(body.username) != body.password:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    return {
        "access_token": f"mock-jwt-{body.username}",
        "token_type": "bearer",
        "username": body.username,
    }
