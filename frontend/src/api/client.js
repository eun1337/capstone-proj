const BASE = '/api';
const REQUEST_TIMEOUT_MS = 25000;
const RETRY_DELAY_MS = 1200;

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function request(path, options = {}, canRetry = true) {
  const token = localStorage.getItem('token');
  let res;
  try {
    res = await fetchWithTimeout(`${BASE}${path}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...options.headers,
      },
    }, REQUEST_TIMEOUT_MS);
  } catch (e) {
    if (canRetry) {
      await sleep(RETRY_DELAY_MS);
      return request(path, options, false);
    }
    throw new Error('일시적으로 서버 응답이 지연되고 있습니다. 잠시 후 다시 시도해주세요.');
  }
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

  getProductOptions: ({ center, sku_id } = {}) =>
    request(`/dashboard/product-options${buildQuery({ center, sku_id })}`),

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

  getDailyProductActivity: ({ center, date, category_large, category_middle, category_small } = {}) =>
    request(`/dashboard/daily/product-activity${buildQuery({ center, date, category_large, category_middle, category_small })}`),

  getDailyTransactions: ({ center, sku_id, date, history_days } = {}) =>
    request(`/dashboard/daily/transactions${buildQuery({ center, sku_id, date, history_days })}`),

  getStatHeatmap: ({ center, metric } = {}) =>
    request(`/model-analysis/stat/heatmap${buildQuery({ center, metric })}`),

  getStatCommonHeatmap: ({ center, metric } = {}) =>
    request(`/model-analysis/stat/common-heatmap${buildQuery({ center, metric })}`),

  getStatCommonWapeBias: ({ center, horizon } = {}) =>
    request(`/model-analysis/stat/common-wape-bias${buildQuery({ center, horizon })}`),

  getStatCoverage: () => request('/model-analysis/stat/coverage'),

  getMlDlModelDetail: ({ model, horizon } = {}) =>
    request(`/model-analysis/mldl/model-detail${buildQuery({ model, horizon })}`),

  getMlDlModelSummary: () => request('/model-analysis/mldl/model-summary'),

  getMlDlImprovementExperiments: () => request('/model-analysis/mldl/improvement-experiments'),

  getFinalKpi: ({ center, horizon, metric, scope } = {}) =>
    request(`/model-analysis/final/kpi${buildQuery({ center, horizon, metric, scope })}`),

  getQaStatVariableEffect: ({ center, horizon } = {}) =>
    request(`/model-analysis/qa/stat-variable-effect${buildQuery({ center, horizon })}`),
};
