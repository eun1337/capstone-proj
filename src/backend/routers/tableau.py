import uuid
import time

import jwt
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter()
security = HTTPBearer(auto_error=False)

# Tableau Connected App 설정값 — 실제 연동 시 환경변수로 교체
TABLEAU_SECRET    = "your-tableau-connected-app-secret-here"
TABLEAU_CLIENT_ID = "your-client-id-here"
TABLEAU_USER      = "tableau-user@example.com"


@router.get("/tableau-token")
def get_tableau_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    now = int(time.time())
    payload = {
        "iss": TABLEAU_CLIENT_ID,
        "exp": now + 300,
        "jti": str(uuid.uuid4()),
        "aud": "tableau",
        "sub": TABLEAU_USER,
        "scp": ["tableau:views:embed"],
    }
    token = jwt.encode(payload, TABLEAU_SECRET, algorithm="HS256")
    return {"token": token, "expires_in": 300}
