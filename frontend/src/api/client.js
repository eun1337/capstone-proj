const BASE = '/api';

async function request(path, options = {}) {
  const token = localStorage.getItem('token');
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: '서버 오류가 발생했습니다.' }));
    throw new Error(err.detail || '요청에 실패했습니다.');
  }
  return res.json();
}

export const api = {
  login: (username, password) =>
    request('/login', { method: 'POST', body: JSON.stringify({ username, password }) }),

  getProducts: ({ center, category, status } = {}) => {
    const params = new URLSearchParams();
    if (center)   params.set('center', center);
    if (category) params.set('category', category);
    if (status)   params.set('status', status);
    const qs = params.toString();
    return request(`/products${qs ? '?' + qs : ''}`);
  },

  getSummary: (center) =>
    request(`/summary${center ? '?center=' + encodeURIComponent(center) : ''}`),

  getTableauToken: () => request('/tableau-token'),
};
