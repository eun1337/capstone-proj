"""
Tableau Connected Apps (직접 신뢰) — JWT 발급 엔드포인트
=========================================================

흐름:
  1. 사용자 → 우리 서비스 로그인 → Bearer 토큰 발급
  2. 프론트엔드 → GET /api/tableau-token (Authorization: Bearer <우리토큰>)
  3. 백엔드 → Tableau JWT 생성 후 { token, expires_in } 반환
  4. 프론트엔드 → <tableau-viz token="..."> 에 주입
             → Tableau 로그인 창 없이 SSO 렌더링

필요 환경변수 (src/backend/.env):
  TABLEAU_CLIENT_ID   — Connected App의 Client ID
  TABLEAU_SECRET_ID   — Connected App Secret의 Secret ID (JWT kid 헤더)
  TABLEAU_SECRET_KEY  — Connected App Secret의 Secret Key (HS256 서명 키)
  TABLEAU_EMBED_USER  — 임베딩에 사용할 Tableau 계정 이메일
                        (해당 뷰 접근 권한이 있어야 함)

라이브러리:
  pip install PyJWT==2.8.0   ← requirements.txt에 이미 포함
"""

import os
import uuid
import time
import logging

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# ── 환경변수에서 Tableau Connected App 설정 로드 ────────────────────────────────
TABLEAU_CLIENT_ID   = os.getenv("TABLEAU_CLIENT_ID", "")
TABLEAU_SECRET_ID   = os.getenv("TABLEAU_SECRET_ID", "")
TABLEAU_SECRET_KEY  = os.getenv("TABLEAU_SECRET_KEY", "")
TABLEAU_EMBED_USER  = os.getenv("TABLEAU_EMBED_USER", "")

# JWT 유효시간(초) — Tableau 권장 최대 10분. 프론트엔드는 만료 1분 전에 갱신한다.
TOKEN_TTL_SECONDS = 540  # 9분

if not all([TABLEAU_CLIENT_ID, TABLEAU_SECRET_ID, TABLEAU_SECRET_KEY, TABLEAU_EMBED_USER]):
    logger.warning(
        "[Tableau] ⚠  Connected App 환경변수 미설정. "
        "TABLEAU_CLIENT_ID / TABLEAU_SECRET_ID / TABLEAU_SECRET_KEY / TABLEAU_EMBED_USER "
        "를 src/backend/.env 에 추가하세요."
    )

# ── 현재 서비스의 mock 사용자→이메일 매핑 ──────────────────────────────────────
# 실제 DB가 생기면 이 dict를 DB 조회로 교체한다.
_USER_EMAIL_MAP: dict[str, str] = {
    # "서비스 username": "Tableau 계정 이메일"
    # 아래 항목은 예시이며, 실제 Tableau 사이트에 등록된 이메일로 교체하세요.
    # "adminid": "admin@yourcompany.com",
    # "user":    "user@yourcompany.com",
}


def _extract_username(credentials: HTTPAuthorizationCredentials | None) -> str:
    """
    우리 서비스 Bearer 토큰에서 사용자명을 추출한다.
    현재는 mock 토큰 형식(mock-jwt-<username>)을 파싱.
    실제 JWT를 사용한다면 jwt.decode() 로 교체하면 된다.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인 후 이용 가능합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    # mock 토큰: "mock-jwt-{username}"
    if token.startswith("mock-jwt-"):
        return token[len("mock-jwt-"):]
    # 실제 JWT라면 여기서 decode 후 sub/username 클레임 반환
    return token


def _resolve_embed_email(username: str) -> str:
    """
    서비스 username → Tableau 계정 이메일 변환.
    매핑 테이블에 없으면 환경변수의 기본 임베딩 계정을 사용한다.
    """
    return _USER_EMAIL_MAP.get(username, TABLEAU_EMBED_USER)


def _build_tableau_jwt(embed_email: str) -> str:
    """
    Tableau Connected Apps 스펙의 JWT 생성.

    Header:
      alg: HS256
      typ: JWT
      iss: <Client ID>   ← Connected App 식별자
      kid: <Secret ID>   ← 어떤 Secret으로 서명했는지 Tableau에게 알림

    Payload:
      iss: <Client ID>
      exp: <Unix 타임스탬프>
      jti: <UUID4>       ← 재전송(replay) 공격 방지용 고유값
      aud: "tableau"     ← 고정값
      sub: <이메일>       ← 임베딩 사용자 (Tableau 사이트에 등록된 이메일)
      scp: ["tableau:views:embed"]  ← 임베딩 전용 최소 권한
    """
    if not all([TABLEAU_CLIENT_ID, TABLEAU_SECRET_ID, TABLEAU_SECRET_KEY, embed_email]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Tableau Connected App 설정이 완료되지 않았습니다. "
                "환경변수(TABLEAU_CLIENT_ID / TABLEAU_SECRET_ID / "
                "TABLEAU_SECRET_KEY / TABLEAU_EMBED_USER)를 확인하세요."
            ),
        )

    now = int(time.time())
    payload = {
        "iss": TABLEAU_CLIENT_ID,
        "exp": now + TOKEN_TTL_SECONDS,
        "jti": str(uuid.uuid4()),
        "aud": "tableau",
        "sub": embed_email,
        "scp": ["tableau:views:embed"],
    }

    token = jwt.encode(
        payload,
        TABLEAU_SECRET_KEY,
        algorithm="HS256",
        headers={
            "kid": TABLEAU_SECRET_ID,
            "iss": TABLEAU_CLIENT_ID,
        },
    )
    return token


@router.get(
    "/tableau-token",
    summary="Tableau Connected Apps SSO 토큰 발급",
    response_description="Tableau 임베딩용 JWT 토큰과 만료 시간(초)",
)
def get_tableau_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    우리 서비스에 로그인한 사용자에게 Tableau 임베딩용 JWT를 발급합니다.

    - 반환된 `token`을 `<tableau-viz token="...">` 속성에 주입하세요.
    - `expires_in` 초 후에 만료되므로, 프론트엔드는 만료 60초 전에 갱신 요청을 보내야 합니다.
    """
    username = _extract_username(credentials)
    embed_email = _resolve_embed_email(username)
    token = _build_tableau_jwt(embed_email)

    logger.info(
        "[Tableau] SSO 토큰 발급 — user=%s, sub=%s, expires_in=%ds",
        username, embed_email, TOKEN_TTL_SECONDS,
    )
    return {"token": token, "expires_in": TOKEN_TTL_SECONDS}


