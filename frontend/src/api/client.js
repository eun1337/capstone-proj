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

function buildQuery(params) {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') qs.set(k, v);
  });
  const s = qs.toString();
  return s ? `?${s}` : '';
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

  getCategories: ({ center } = {}) => {
    const params = new URLSearchParams();
    if (center) params.set('center', center);
    const qs = params.toString();
    return request(`/dashboard/categories${qs ? '?' + qs : ''}`);
  },

  getDashboardProducts: ({ center, category_large, category_middle, category_small, search } = {}) => {
    const params = new URLSearchParams();
    if (center)          params.set('center', center);
    if (category_large)  params.set('category_large', category_large);
    if (category_middle) params.set('category_middle', category_middle);
    if (category_small)  params.set('category_small', category_small);
    if (search)           params.set('search', search);
    const qs = params.toString();
    return request(`/dashboard/products${qs ? '?' + qs : ''}`);
  },

  getDashboardSummary: ({ center, week_st, category_large, category_middle, category_small, sku_id } = {}) => {
    const params = new URLSearchParams();
    if (center)          params.set('center', center);
    if (week_st)          params.set('week_st', week_st);
    if (category_large)  params.set('category_large', category_large);
    if (category_middle) params.set('category_middle', category_middle);
    if (category_small)  params.set('category_small', category_small);
    if (sku_id)           params.set('sku_id', sku_id);
    const qs = params.toString();
    return request(`/dashboard/summary${qs ? '?' + qs : ''}`);
  },

  getDashboardForecast: ({ center, sku_id, week_st, history_weeks } = {}) => {
    const params = new URLSearchParams();
    if (center)        params.set('center', center);
    if (sku_id)         params.set('sku_id', sku_id);
    if (week_st)        params.set('week_st', week_st);
    if (history_weeks) params.set('history_weeks', history_weeks);
    const qs = params.toString();
    return request(`/dashboard/forecast${qs ? '?' + qs : ''}`);
  },

  getForecastProducts: ({ center, week_st, category_large, category_middle, category_small, option_code, top } = {}) => {
    const params = new URLSearchParams();
    if (center)          params.set('center', center);
    if (week_st)          params.set('week_st', week_st);
    if (category_large)  params.set('category_large', category_large);
    if (category_middle) params.set('category_middle', category_middle);
    if (category_small)  params.set('category_small', category_small);
    if (option_code)      params.set('option_code', option_code);
    if (top)              params.set('top', top);
    const qs = params.toString();
    return request(`/dashboard/forecast-products${qs ? '?' + qs : ''}`);
  },

  // 상품명만으로는 절대 합치지 않는 안전한 identity(상품명+규격+KAN_소분류)로 판별된
  // "같은 상품의 다른 판매단위(EA/BX/CS)" SKU 목록.
  getProductOptions: ({ center, sku_id } = {}) =>
    request(`/dashboard/product-options${buildQuery({ center, sku_id })}`),

  // 메인 "수요예측 추이" 차트의 집계(카테고리/센터 범위, SKU 미선택) 모드 전용 —
  // weekly_demand/forecast_2024을 같은 단위 SKU들에 대해 그대로 합산만 한다.
  getDemandTrend: ({ center, option_code, category_large, category_middle, category_small, week_st, history_weeks } = {}) =>
    request(`/dashboard/demand-trend${buildQuery({ center, option_code, category_large, category_middle, category_small, week_st, history_weeks })}`),

  getDashboardInventory: ({ center, sku_id, week_st, return_policy } = {}) => {
    const params = new URLSearchParams();
    if (center)         params.set('center', center);
    if (sku_id)          params.set('sku_id', sku_id);
    if (week_st)         params.set('week_st', week_st);
    if (return_policy)  params.set('return_policy', return_policy);
    const qs = params.toString();
    return request(`/dashboard/inventory${qs ? '?' + qs : ''}`);
  },

  getDashboardTransactions: ({ center, sku_id, week_st, history_weeks } = {}) => {
    const params = new URLSearchParams();
    if (center)        params.set('center', center);
    if (sku_id)         params.set('sku_id', sku_id);
    if (week_st)        params.set('week_st', week_st);
    if (history_weeks) params.set('history_weeks', history_weeks);
    const qs = params.toString();
    return request(`/dashboard/transactions${qs ? '?' + qs : ''}`);
  },

  // ── 운영 인사이트 (/dashboard/insights/*) ──────────────────────
  getInsightCategorySales: ({ center, week_st, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/insights/category-sales${buildQuery({ center, week_st, category_large, category_middle, category_small })}`),

  getInsightRegionSales: ({ center, week_st, sido, sigungu, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/insights/region-sales${buildQuery({ center, week_st, sido, sigungu, category_large, category_middle, category_small })}`),

  getInsightInventoryShortage: ({ center, week_st, category_large, category_middle, category_small, option_code } = {}) =>
    request(`/dashboard/insights/inventory-shortage${buildQuery({ center, week_st, category_large, category_middle, category_small, option_code })}`),

  getInsightSalesSurge: ({ center, week_st, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/insights/sales-surge${buildQuery({ center, week_st, category_large, category_middle, category_small })}`),

  getInsightReturns: ({ center, week_st, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/insights/returns${buildQuery({ center, week_st, category_large, category_middle, category_small })}`),

  // ── 일간 운영 데이터 (/dashboard/daily/*) — 운영실적/재고·입출고 거래는 일간, AI 예측(h1/h2/h4)은
  // 별도로 기존 주간 endpoint(getDashboardForecast/getForecastProducts)를 그대로 사용한다.
  getDailySummary: ({ center, date, category_large, category_middle, category_small, sku_id } = {}) =>
    request(`/dashboard/daily/summary${buildQuery({ center, date, category_large, category_middle, category_small, sku_id })}`),

  getDailyCategorySales: ({ center, date, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/daily/category-sales${buildQuery({ center, date, category_large, category_middle, category_small })}`),

  getDailyRegionSales: ({ center, date, sido, sigungu, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/daily/region-sales${buildQuery({ center, date, sido, sigungu, category_large, category_middle, category_small })}`),

  getDailySalesSurge: ({ center, date, category_large, category_middle, category_small, option_code } = {}) =>
    request(`/dashboard/daily/sales-surge${buildQuery({ center, date, category_large, category_middle, category_small, option_code })}`),

  getDailyReturns: ({ center, date, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/daily/returns${buildQuery({ center, date, category_large, category_middle, category_small })}`),

  // 대시보드 KPI modal(판매금액/순판매금액/판매 SKU 수/반품률) 4개가 공유하는 SKU별 당일
  // 활동 목록 — 각 modal이 이 하나의 목록을 받아 자신에게 맞는 조건/정렬만 클라이언트에서 적용한다.
  getDailyProductActivity: ({ center, date, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/daily/product-activity${buildQuery({ center, date, category_large, category_middle, category_small })}`),

  getDailyTransactions: ({ center, sku_id, date, history_days } = {}) =>
    request(`/dashboard/daily/transactions${buildQuery({ center, sku_id, date, history_days })}`),
};
