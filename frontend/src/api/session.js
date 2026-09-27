// 로그인 세션(localStorage) 공용 헬퍼 — 토큰 만료 시각은 백엔드 expires_in(3시간) 기준으로 저장한다.
const KEYS = ['token', 'username', 'token_expires_at'];

export function saveSession({ access_token, username, expires_in }) {
  localStorage.setItem('token', access_token);
  localStorage.setItem('username', username);
  localStorage.setItem('token_expires_at', String(Date.now() + expires_in * 1000));
}

export function clearSession() {
  KEYS.forEach((k) => localStorage.removeItem(k));
}

export function hasValidSession() {
  const token = localStorage.getItem('token');
  const expiresAt = Number(localStorage.getItem('token_expires_at'));
  if (token && expiresAt && Date.now() < expiresAt) return true;
  clearSession();
  return false;
}
