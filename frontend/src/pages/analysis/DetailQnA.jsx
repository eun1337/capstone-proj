import { useEffect, useState } from 'react';
import { api } from '../../api/client.js';
import EChart from '../../components/charts/EChart.jsx';
import './analysis.css';
import './DetailQnA.css';

const CENTERS = [
  { value: 'ALL', label: 'ALL' },
  { value: 'A', label: 'A' },
  { value: 'B', label: 'B' },
];
const HORIZONS = [
  { value: 'h1', label: 'h1' },
  { value: 'h2', label: 'h2' },
  { value: 'h4', label: 'h4' },
];
const MODELS = [
  { value: 'compare', label: '비교 (SARIMA / H-LGBM)' },
  { value: 'SARIMA', label: 'SARIMA' },
  { value: 'Hurdle-LightGBM', label: 'H-LGBM' },
];
const QUESTIONS = [
  { key: 'error', label: '언제 오차가 컸나?' },
  { key: 'stat', label: '통계 변수 효과는?' },
  { key: 'coverage', label: '모델 적용 범위는?' },
  { key: 'sku', label: '어떤 SKU에서 달랐나?' },
  { key: 'param', label: '파라미터는 어떻게 정했나?' },
];
const FILTER_VISIBILITY = {
  sku: ['center', 'horizon', 'model', 'week', 'sku'],
  error: ['center', 'horizon', 'model', 'week'],
  stat: ['center', 'horizon'],
  coverage: ['center', 'horizon'],
  param: ['model'],
};
const SCOPE_LABEL_KO = { model_fit: '실제 SARIMA 모델을 적합해 예측', fallback: 'SARIMA 적합 실패 시 대체 방식으로 예측', cold_start: '이력 부족으로 초기값 기반 예측' };

const STAT_COLOR = '#3b82f6';
const ML_COLOR = '#8b5cf6';
const ACTUAL_COLOR = '#1e293b';
const TOOLTIP_BASE = { appendTo: 'body', confine: true, extraCssText: 'z-index:99999;' };

function fmtNum(v, digits = 1) {
  return v === null || v === undefined || Number.isNaN(v) ? '-' : v.toFixed(digits);
}
function fmtInt(v) {
  return v === null || v === undefined ? '-' : Math.round(v).toLocaleString();
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function robustBound(values) {
  const abs = values.map((v) => Math.abs(v)).filter((v) => Number.isFinite(v));
  if (abs.length === 0) return 1;
  const max = Math.max(...abs);
  if (abs.length <= 4) return max || 1;
  const med = median(abs);
  const mad = median(abs.map((v) => Math.abs(v - med))) || med * 0.5 || 1;
  return Math.min(max, Math.max(med + mad * 4, med, 0.001));
}

export default function DetailQnA() {
  const [activeQuestion, setActiveQuestion] = useState('sku');
  const [center, setCenter] = useState('ALL');
  const [horizon, setHorizon] = useState('h1');
  const [model, setModel] = useState('compare');
  const [week, setWeek] = useState('');
  const [skuSearchInput, setSkuSearchInput] = useState('');
  const [skuSearch, setSkuSearch] = useState('');
  const [selectedSku, setSelectedSku] = useState(null);                      

  const [weeks, setWeeks] = useState([]);
  const [scatter, setScatter] = useState(null);
  const [scatterError, setScatterError] = useState(null);
  const [skuDetail, setSkuDetail] = useState(null);
  const [skuDetailError, setSkuDetailError] = useState(null);
  const [skuWeekly, setSkuWeekly] = useState(null);
  const [skuWeeklyError, setSkuWeeklyError] = useState(null);
  const [weeklyError, setWeeklyError] = useState(null);
  const [weeklyErrorError, setWeeklyErrorError] = useState(null);
  const [coverage, setCoverage] = useState(null);
  const [coverageError, setCoverageError] = useState(null);
  const [statVariable, setStatVariable] = useState(null);
  const [statVariableError, setStatVariableError] = useState(null);
  const [paramSummary, setParamSummary] = useState(null);
  const [paramSummaryError, setParamSummaryError] = useState(null);
  const [modalModel, setModalModel] = useState(null);

  useEffect(() => { api.getQaWeeks().then((d) => setWeeks(d.weeks)).catch(() => {}); }, []);

  useEffect(() => {
    const t = setTimeout(() => setSkuSearch(skuSearchInput.trim()), 300);
    return () => clearTimeout(t);
  }, [skuSearchInput]);

  useEffect(() => {
    let ignore = false;
    setScatterError(null);
    api.getQaSkuScatter({ center, horizon, sku_search: skuSearch || undefined })
      .then((d) => { if (!ignore) { setScatter(d); setSelectedSku((prev) => prev || (d.points[0] ? { sku_id: d.points[0].sku_id, center: d.points[0].center } : null)); } })
      .catch((e) => { if (!ignore) { setScatterError(e.message); setScatter(null); } });
    return () => { ignore = true; };
  }, [center, horizon, skuSearch]);

  useEffect(() => {
    if (!selectedSku) { setSkuDetail(null); return; }
    let ignore = false;
    setSkuDetailError(null);
    api.getQaSkuDetail({ sku_id: selectedSku.sku_id, center: selectedSku.center, horizon })
      .then((d) => { if (!ignore) setSkuDetail(d); })
      .catch((e) => { if (!ignore) { setSkuDetailError(e.message); setSkuDetail(null); } });
    return () => { ignore = true; };
  }, [selectedSku, horizon]);

  useEffect(() => {
    if (!selectedSku) { setSkuWeekly(null); return; }
    let ignore = false;
    setSkuWeeklyError(null);
    api.getQaSkuWeekly({ sku_id: selectedSku.sku_id, center: selectedSku.center, horizon })
      .then((d) => { if (!ignore) setSkuWeekly(d); })
      .catch((e) => { if (!ignore) { setSkuWeeklyError(e.message); setSkuWeekly(null); } });
    return () => { ignore = true; };
  }, [selectedSku, horizon]);

  useEffect(() => {
    let ignore = false;
    setWeeklyErrorError(null);
    api.getQaWeeklyError({ center, horizon })
      .then((d) => { if (!ignore) setWeeklyError(d); })
      .catch((e) => { if (!ignore) { setWeeklyErrorError(e.message); setWeeklyError(null); } });
    return () => { ignore = true; };
  }, [center, horizon]);

  useEffect(() => {
    let ignore = false;
    setCoverageError(null);
    const params = selectedSku
      ? { center, horizon, sku_id: selectedSku.sku_id, sku_center: selectedSku.center }
      : { center, horizon };
    api.getQaCoverage(params)
      .then((d) => { if (!ignore) setCoverage(d); })
      .catch((e) => { if (!ignore) { setCoverageError(e.message); setCoverage(null); } });
    return () => { ignore = true; };
  }, [center, horizon, selectedSku]);

  useEffect(() => {
    let ignore = false;
    setStatVariableError(null);
    api.getQaStatVariableEffect({ center, horizon })
      .then((d) => { if (!ignore) setStatVariable(d); })
      .catch((e) => { if (!ignore) { setStatVariableError(e.message); setStatVariable(null); } });
    return () => { ignore = true; };
  }, [center, horizon]);

  useEffect(() => {
    let ignore = false;
    setParamSummaryError(null);
    api.getQaParameterSummary({ models: model === 'compare' ? undefined : model })
      .then((d) => { if (!ignore) setParamSummary(d); })
      .catch((e) => { if (!ignore) { setParamSummaryError(e.message); setParamSummary(null); } });
    return () => { ignore = true; };
  }, [model]);

  function handleSelectSku(sku_id, skuCenter) {
    setSelectedSku({ sku_id, center: skuCenter });
  }

  function handleSelectWeek(w) {
    setWeek(w);
  }

  const visibleFilters = FILTER_VISIBILITY[activeQuestion];
  let main;
  let secondary;
  if (activeQuestion === 'error') {
    main = <MainWeeklyErrorCard data={weeklyError} error={weeklyErrorError} model={model} week={week} onSelectWeek={handleSelectWeek} />;
    secondary = [
      <Top5ErrorWeeksCard key="top5" data={weeklyError} week={week} onSelectWeek={handleSelectWeek} />,
      <SelectedWeekDetailCard key="weekdetail" data={weeklyError} week={week} />,
    ];
  } else if (activeQuestion === 'stat') {
    main = <MainStatVariableCard data={statVariable} error={statVariableError} />;
    secondary = [<StatSummaryCard key="statsum" data={statVariable} />];
  } else if (activeQuestion === 'coverage') {
    main = <MainCoverageCard data={coverage} error={coverageError} />;
    secondary = [<CoverageDefinitionCard key="def" />];
  } else if (activeQuestion === 'param') {
    main = <MainParamSummaryCard data={paramSummary} error={paramSummaryError} onOpenModel={setModalModel} />;
    secondary = (paramSummary?.models || []).map((m) => (
      <ParamHighlightCard key={m.model} model={m} onOpen={() => setModalModel(m)} />
    ));
  } else {
    main = <MainScatterCard data={scatter} error={scatterError} selectedSku={selectedSku} onSelect={handleSelectSku} />;
    secondary = [
      <SkuDetailCard key="skudetail" data={skuDetail} error={skuDetailError} />,
      <SkuWeeklyPredictCard key="skuweekly" data={skuWeekly} error={skuWeeklyError} model={model} week={week} selectedSku={selectedSku} onSelectWeek={handleSelectWeek} />,
    ];
  }

  return (
    <div className="qa-page">
      <div className="az-page-hd">
        <div>
          <h2>05 상세 분석 / Q&A</h2>
          <p>질문을 선택하면 메인 분석 영역이 그 질문 중심으로 바뀝니다</p>
        </div>
      </div>

      <div className="qa-questions">
        {QUESTIONS.map((q) => (
          <button
            key={q.key}
            className={`qa-question-btn ${activeQuestion === q.key ? 'active' : ''}`}
            onClick={() => setActiveQuestion(q.key)}
          >
            {q.label}
          </button>
        ))}
      </div>

      <div className="az-filter-bar qa-filter-bar">
        {visibleFilters.includes('center') && (
          <div className="az-filter-group">
            <span className="az-filter-label">Center</span>
            <select className="az-filter-select" value={center} onChange={(e) => setCenter(e.target.value)}>
              {CENTERS.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
            </select>
          </div>
        )}
        {visibleFilters.includes('horizon') && (
          <div className="az-filter-group">
            <span className="az-filter-label">Horizon</span>
            <select className="az-filter-select" value={horizon} onChange={(e) => setHorizon(e.target.value)}>
              {HORIZONS.map((h) => <option key={h.value} value={h.value}>{h.label}</option>)}
            </select>
          </div>
        )}
        {visibleFilters.includes('model') && (
          <div className="az-filter-group">
            <span className="az-filter-label">Model</span>
            <select className="az-filter-select" value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        )}
        {visibleFilters.includes('week') && (
          <div className="az-filter-group">
            <span className="az-filter-label">Week</span>
            <select className="az-filter-select" value={week} onChange={(e) => handleSelectWeek(e.target.value)}>
              <option value="">2024 전체</option>
              {weeks.map((w) => <option key={w} value={w}>{w}</option>)}
            </select>
          </div>
        )}
        {visibleFilters.includes('sku') && (
          <div className="az-filter-group qa-sku-search">
            <span className="az-filter-label">SKU 검색</span>
            <input
              className="qa-search-input"
              value={skuSearchInput}
              onChange={(e) => setSkuSearchInput(e.target.value)}
              placeholder="SKU명을 입력하세요..."
            />
          </div>
        )}
      </div>

      <div className="qa-charts">
        <div className="qa-main-area">{main}</div>
        <div className="qa-secondary-area">{secondary}</div>
      </div>

      {modalModel && <ParamDetailModal model={modalModel} onClose={() => setModalModel(null)} />}
    </div>
  );
}

function MainScatterCard({ data, error, selectedSku, onSelect }) {
  return (
    <div className="az-card qa-card qa-card-main">
      <div className="az-card-hd">
        <h3>Actual demand vs Model advantage Scatter</h3>
        <p>SKU별 실제 수요와 모델 우세 정도 비교 {data ? `· ${data.total.toLocaleString()}개 SKU` : ''}</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && data.points.length === 0 && <div className="az-hint">조건에 맞는 SKU가 없습니다.</div>}
      {!error && data && data.points.length > 0 && (
        <EChart
          fill
          renderer="canvas"
          option={buildScatterOption(data, selectedSku)}
          onEvents={{ click: (p) => { if (p.data?.sku_id) onSelect(p.data.sku_id, p.data.center); } }}
        />
      )}
    </div>
  );
}

function buildScatterOption(data, selectedSku) {
  const bound = robustBound(data.points.map((p) => p.diff));
  const clampY = (v) => Math.max(-bound, Math.min(bound, v));
  const points = data.points.map((p) => ({
    value: [p.actual_sum, clampY(p.diff)], sku_id: p.sku_id, center: p.center, raw: p,
    itemStyle: { color: p.diff >= 0 ? ML_COLOR : '#94a3b8', opacity: 0.32 },
    symbolSize: 6,
  }));
  const sel = selectedSku && data.points.find((p) => p.sku_id === selectedSku.sku_id && p.center === selectedSku.center);
  if (sel) {
    points.push({
      value: [sel.actual_sum, clampY(sel.diff)], sku_id: sel.sku_id, center: sel.center, raw: sel,
      itemStyle: { color: sel.diff >= 0 ? ML_COLOR : '#64748b', opacity: 1, borderColor: '#1e293b', borderWidth: 2 },
      symbolSize: 16, label: { show: true, formatter: () => sel.sku_id, position: 'top', fontSize: 11, fontWeight: 700, color: '#1e293b' },
    });
  }
  return {
    grid: { left: 52, right: 16, top: 10, bottom: 34 },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => `${p.data.sku_id} (${p.data.center})<br/>실제 수요: ${fmtInt(p.data.raw.actual_sum)}<br/>SARIMA WAPE: ${fmtNum(p.data.raw.stat_wape)}%<br/>H-LGBM WAPE: ${fmtNum(p.data.raw.ml_wape)}%<br/>차이: ${fmtNum(p.data.raw.diff)}%p`,
    },
    xAxis: {
      type: 'log', name: 'actual_sum (log scale)', nameLocation: 'middle', nameGap: 24,
      nameTextStyle: { fontSize: 11, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 11 },
      axisLine: { lineStyle: { color: '#e2e8f0' } }, splitLine: { show: false },
    },
    yAxis: {
      type: 'value', name: 'SARIMA WAPE − H-LGBM WAPE(%p)', min: -bound, max: bound,
      nameTextStyle: { fontSize: 10, color: '#94a3b8' },
      axisLabel: { color: '#64748b', fontSize: 11, formatter: (v) => v.toFixed(1) },
      splitLine: { lineStyle: { color: '#f1f5f9' } },
    },
    series: [{
      type: 'scatter', data: points, large: true, largeThreshold: 500,
      markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#94a3b8', type: 'dashed' }, data: [{ yAxis: 0 }] },
    }],
  };
}

function SkuDetailCard({ data, error }) {
  return (
    <div className="az-card qa-card">
      <div className="az-card-hd">
        <h3>선택 SKU 상세</h3>
        <p>선택 SKU의 성능 비교</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">SKU를 선택하세요</div>}
      {!error && data && (
        <div className="qa-sku-detail-grid">
          <div><span className="qa-kv-key">SKU</span><span className="qa-kv-val">{data.sku_id}</span></div>
          <div><span className="qa-kv-key">Center</span><span className="qa-kv-val">{data.center}</span></div>
          <div><span className="qa-kv-key">actual_sum</span><span className="qa-kv-val">{fmtInt(data.actual_sum)}</span></div>
          <div><span className="qa-kv-key">SARIMA WAPE</span><span className="qa-kv-val">{fmtNum(data.stat_wape)}%</span></div>
          <div><span className="qa-kv-key">H-LGBM WAPE</span><span className="qa-kv-val">{fmtNum(data.ml_wape)}%</span></div>
          <div><span className="qa-kv-key">winner</span><span className={`qa-winner-badge ${data.winner === 'Hurdle-LightGBM' ? 'is-ml' : 'is-stat'}`}>{data.winner === 'Hurdle-LightGBM' ? 'H-LGBM 우세' : 'SARIMA 우세'}</span></div>
          <div><span className="qa-kv-key">수요구간</span><span className="qa-kv-val">{data.quartile_label}</span></div>
        </div>
      )}
    </div>
  );
}

function SkuWeeklyPredictCard({ data, error, model, week, selectedSku, onSelectWeek }) {
  return (
    <div className="az-card qa-card">
      <div className="az-card-hd">
        <h3>주간 Actual vs Prediction</h3>
        <p>선택 SKU의 2024 주차별 예측 비교</p>
      </div>
      {!selectedSku && <div className="az-hint">SKU를 선택하세요</div>}
      {selectedSku && error && <div className="az-hint az-hint-error">{error}</div>}
      {selectedSku && !error && !data && <div className="az-hint">불러오는 중...</div>}
      {selectedSku && !error && data && (
        <EChart fill option={buildSkuWeeklyOption(data, model, week)} onEvents={{ click: (p) => p.data?.target_week && onSelectWeek(p.data.target_week) }} />
      )}
    </div>
  );
}

function buildSkuWeeklyOption(data, model, week) {
  const cats = data.weeks.map((w) => w.week_index);
  const series = [{
    name: 'Actual', type: 'line', showSymbol: false, lineStyle: { width: 2, color: ACTUAL_COLOR }, itemStyle: { color: ACTUAL_COLOR },
    data: data.weeks.map((w) => ({ value: w.actual, target_week: w.target_week })),
  }];
  if (model !== 'Hurdle-LightGBM') {
    series.push({
      name: 'SARIMA', type: 'line', showSymbol: false, lineStyle: { width: 1.5, color: STAT_COLOR, type: 'dashed' }, itemStyle: { color: STAT_COLOR },
      data: data.weeks.map((w) => ({ value: w.stat_prediction, target_week: w.target_week })),
    });
  }
  if (model !== 'SARIMA') {
    series.push({
      name: 'H-LGBM', type: 'line', showSymbol: false, lineStyle: { width: 1.5, color: ML_COLOR, type: 'dashed' }, itemStyle: { color: ML_COLOR },
      data: data.weeks.map((w) => ({ value: w.ml_prediction, target_week: w.target_week })),
    });
  }
  const selIdx = week ? data.weeks.findIndex((w) => w.target_week === week) : -1;
  if (selIdx >= 0) {
    series[0].markLine = { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#1e293b', type: 'dashed' }, data: [{ xAxis: selIdx }] };
  }
  return {
    grid: { left: 40, right: 12, top: 22, bottom: 22 },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: '#475569' } },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis', formatter: (ps) => `주차 ${ps[0].axisValue}${ps[0].data.target_week ? ` (${ps[0].data.target_week})` : ''}<br/>` + ps.map((p) => `${p.seriesName}: ${p.data.value === null || p.data.value === undefined ? '-' : fmtNum(p.data.value)}`).join('<br/>') },
    xAxis: { type: 'category', data: cats, name: '주차', nameLocation: 'middle', nameGap: 20, nameTextStyle: { fontSize: 10, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 10.5 }, axisLine: { lineStyle: { color: '#e2e8f0' } } },
    yAxis: { type: 'value', name: '수요량', nameTextStyle: { fontSize: 10, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series,
  };
}

function MainWeeklyErrorCard({ data, error, model, week, onSelectWeek }) {
  return (
    <div className="az-card qa-card qa-card-main">
      <div className="az-card-hd">
        <h3>언제 오차가 컸나?</h3>
        <p>주차별 WAPE 추이 · 굵은 점선 = 선택 주차, 빨강 = 오차 TOP5</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <EChart fill option={buildWeeklyErrorOption(data, model, week)} onEvents={{ click: (p) => p.data?.target_week && onSelectWeek(p.data.target_week) }} />
      )}
    </div>
  );
}

function buildWeeklyErrorOption(data, model, week) {
  const cats = data.points.map((p) => p.week_index);
  const topSet = new Set(data.top_error_weeks);
  const series = [];
  if (model !== 'Hurdle-LightGBM') {
    series.push({
      name: 'SARIMA', type: 'line', showSymbol: false, lineStyle: { width: 2, color: STAT_COLOR }, itemStyle: { color: STAT_COLOR },
      data: data.points.map((p) => ({ value: p.stat_wape, target_week: p.target_week, itemStyle: topSet.has(p.target_week) ? { color: '#dc2626' } : undefined })),
    });
  }
  if (model !== 'SARIMA') {
    series.push({
      name: 'H-LGBM', type: 'line', showSymbol: false, lineStyle: { width: 2, color: ML_COLOR }, itemStyle: { color: ML_COLOR },
      data: data.points.map((p) => ({ value: p.ml_wape, target_week: p.target_week, itemStyle: topSet.has(p.target_week) ? { color: '#dc2626' } : undefined })),
    });
  }
  const selIdx = week ? data.points.findIndex((p) => p.target_week === week) : -1;
  if (selIdx >= 0 && series.length) {
    series[0].markLine = { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#1e293b', type: 'dashed' }, data: [{ xAxis: selIdx }] };
  }
  return {
    grid: { left: 44, right: 16, top: 26, bottom: 26 },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 12, color: '#475569' } },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis', formatter: (ps) => `주차 ${ps[0].axisValue}${ps[0].data.target_week ? ` (${ps[0].data.target_week})` : ''}<br/>` + ps.map((p) => `${p.seriesName}: ${fmtNum(p.data.value)}%`).join('<br/>') },
    xAxis: { type: 'category', data: cats, name: '주차', nameLocation: 'middle', nameGap: 22, nameTextStyle: { fontSize: 11, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 11 }, axisLine: { lineStyle: { color: '#e2e8f0' } } },
    yAxis: { type: 'value', name: 'WAPE(%)', nameTextStyle: { fontSize: 11, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 11.5 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series,
  };
}

function Top5ErrorWeeksCard({ data, week, onSelectWeek }) {
  return (
    <div className="az-card qa-card">
      <div className="az-card-hd">
        <h3>오차 큰 주 TOP5</h3>
        <p>SARIMA·H-LGBM 평균 WAPE 기준</p>
      </div>
      {!data && <div className="az-hint">불러오는 중...</div>}
      {data && (
        <ol className="qa-top5-list">
          {data.top_error_weeks.map((w, i) => {
            const pt = data.points.find((p) => p.target_week === w);
            return (
              <li key={w} className={`qa-top5-item ${week === w ? 'is-selected' : ''}`} onClick={() => onSelectWeek(w)}>
                <span className="qa-top5-rank">{i + 1}</span>
                <span className="qa-top5-week">{w}</span>
                <span className="qa-top5-val">SARIMA {fmtNum(pt?.stat_wape)}% · H-LGBM {fmtNum(pt?.ml_wape)}%</span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

function SelectedWeekDetailCard({ data, week }) {
  const pt = data && week ? data.points.find((p) => p.target_week === week) : null;
  return (
    <div className="az-card qa-card">
      <div className="az-card-hd">
        <h3>선택 주차 상세</h3>
      </div>
      {!week && <div className="az-hint">주차를 선택하세요</div>}
      {week && !pt && <div className="az-hint">해당 주차 데이터가 없습니다.</div>}
      {pt && (
        <div className="qa-sku-detail-grid">
          <div><span className="qa-kv-key">주차</span><span className="qa-kv-val">{pt.target_week} (W{pt.week_index})</span></div>
          <div />
          <div><span className="qa-kv-key">SARIMA WAPE</span><span className="qa-kv-val">{fmtNum(pt.stat_wape)}%</span></div>
          <div><span className="qa-kv-key">H-LGBM WAPE</span><span className="qa-kv-val">{fmtNum(pt.ml_wape)}%</span></div>
          <div><span className="qa-kv-key">SARIMA Bias</span><span className="qa-kv-val">{fmtNum(pt.stat_bias)}%</span></div>
          <div><span className="qa-kv-key">H-LGBM Bias</span><span className="qa-kv-val">{fmtNum(pt.ml_bias)}%</span></div>
        </div>
      )}
    </div>
  );
}

function MainStatVariableCard({ data, error }) {
  return (
    <div className="az-card qa-card qa-card-main">
      <div className="az-card-hd">
        <h3>통계 변수 효과</h3>
        <p>외생변수 설정별 WAPE 변화 (낮을수록 우수) · ARIMA-S0 → S1(경제) → S2(COVID) → S3(공휴일) → S4(전체) → SARIMA → SARIMAX</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && <EChart fill option={buildStatVariableOption(data)} />}
    </div>
  );
}

function buildStatVariableOption(data) {
  const bound = robustBound(data.entries.map((e) => e.wape));
  return {
    grid: { left: 48, right: 16, top: 16, bottom: 44 },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => `${p.data.raw.label}${p.data.raw.is_extreme ? ' (발산)' : ''}<br/>WAPE: ${fmtNum(p.data.raw.wape, 2)}%<br/>Bias: ${fmtNum(p.data.raw.bias, 2)}%<br/>MAE: ${fmtNum(p.data.raw.mae, 3)}`,
    },
    xAxis: {
      type: 'category', data: data.entries.map((e) => e.label),
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11.5, interval: 0, rotate: 20 },
    },
    yAxis: { type: 'value', max: bound, name: 'WAPE(%)', nameTextStyle: { fontSize: 11, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 11.5, formatter: (v) => v.toFixed(1) }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{
      type: 'bar', barMaxWidth: 46,
      data: data.entries.map((e) => ({
        value: Math.min(e.wape, bound), raw: e,
        itemStyle: { color: e.label === 'SARIMA' ? '#1d4ed8' : e.is_extreme ? '#fca5a5' : '#93c5fd' },
        label: { show: true, position: 'top', formatter: () => (e.wape > bound ? `${e.wape.toExponential(1)}▲` : fmtNum(e.wape, 1)), fontSize: 11, color: '#475569' },
      })),
    }],
  };
}

function StatSummaryCard({ data }) {
  return (
    <div className="az-card qa-card">
      <div className="az-card-hd">
        <h3>요약</h3>
        <p>발산 모델(ARIMAX-S1/S4) 제외 기준</p>
      </div>
      {!data && <div className="az-hint">불러오는 중...</div>}
      {data && (() => {
        const stable = data.entries.filter((e) => !e.is_extreme);
        const best = stable.length ? stable.reduce((a, b) => (b.wape < a.wape ? b : a)) : null;
        const worst = stable.length ? stable.reduce((a, b) => (b.wape > a.wape ? b : a)) : null;
        const sarima = data.entries.find((e) => e.label === 'SARIMA');
        return (
          <div className="qa-sku-detail-grid">
            {best && <div><span className="qa-kv-key">최저 WAPE</span><span className="qa-kv-val">{best.label} ({fmtNum(best.wape)}%)</span></div>}
            {worst && <div><span className="qa-kv-key">최고 WAPE</span><span className="qa-kv-val">{worst.label} ({fmtNum(worst.wape)}%)</span></div>}
            {sarima && <div><span className="qa-kv-key">최종 선정(SARIMA)</span><span className="qa-kv-val">{fmtNum(sarima.wape)}%</span></div>}
          </div>
        );
      })()}
    </div>
  );
}

function MainCoverageCard({ data, error }) {
  return (
    <div className="az-card qa-card qa-card-main">
      <div className="az-card-hd">
        <h3>모델 적용 범위</h3>
        <p>{data?.source === 'sku' ? '선택 SKU의 SARIMA 적용 범위' : 'Center/Horizon 기준 SARIMA 적용 범위'}</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <div className="qa-coverage-main">
          <div className="qa-coverage-chart"><EChart fill option={buildCoverageOption(data)} /></div>
          <div className="qa-coverage-table-wrap">
            <div className="qa-table-scroll">
              <table className="qa-param-table">
                <thead><tr><th>구분</th><th>정의</th><th>건수</th><th>비율</th></tr></thead>
                <tbody>
                  {data.entries.map((e) => (
                    <tr key={e.scope}>
                      <td className="qa-param-model">{e.label}</td>
                      <td>{SCOPE_LABEL_KO[e.scope] || '-'}</td>
                      <td>{fmtInt(e.count)}</td>
                      <td>{fmtNum(e.ratio)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="qa-coverage-note">{data.note}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function buildCoverageOption(data) {
  const colors = { model_fit: '#2563eb', fallback: '#93c5fd', cold_start: '#e2e8f0' };
  return {
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => `${p.data.label}<br/>${SCOPE_LABEL_KO[p.data.scope] || ''}<br/>${fmtInt(p.data.value)}건 (${fmtNum(p.percent)}%)`,
    },
    legend: { show: false },
    series: [{
      type: 'pie', radius: ['48%', '75%'], center: ['50%', '50%'],
      label: { formatter: (p) => `${p.name}\n${fmtNum(p.percent)}%`, fontSize: 11.5, color: '#475569' },
      labelLine: { length: 6, length2: 6 },
      data: data.entries.map((e) => ({ name: e.label, value: e.count, scope: e.scope, itemStyle: { color: colors[e.scope] || '#cbd5e1' } })),
    }],
  };
}

function CoverageDefinitionCard() {
  return (
    <div className="az-card qa-card">
      <div className="az-card-hd">
        <h3>적용 방식 정의</h3>
        <p>SARIMA vs H-LGBM 커버리지 개념 차이</p>
      </div>
      <ul className="qa-def-list">
        <li><b>Model-fit</b>{SCOPE_LABEL_KO.model_fit}</li>
        <li><b>Fallback</b>{SCOPE_LABEL_KO.fallback}</li>
        <li><b>Cold-start</b>{SCOPE_LABEL_KO.cold_start}</li>
      </ul>
      <p className="qa-coverage-note">H-LGBM은 별도 Fallback/Cold-start 구분 없이 항상 Model-fit로 적용됩니다.</p>
    </div>
  );
}

function MainParamSummaryCard({ data, error, onOpenModel }) {
  return (
    <div className="az-card qa-card qa-card-main">
      <div className="az-card-hd">
        <h3>파라미터 요약</h3>
        <p>모델별 핵심 파라미터·선정 기준</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && data.models.length === 0 && <div className="az-hint">데이터가 없습니다.</div>}
      {!error && data && data.models.length > 0 && (
        <div className="qa-table-scroll">
          <table className="qa-param-table">
            <thead>
              <tr><th>모델</th><th>구분</th><th>탐색 범위/규칙</th><th>선택값/고정값</th><th>선정 기준</th></tr>
            </thead>
            <tbody>
              {data.models.map((m) => (
                m.rows.map((r, i) => (
                  <tr key={`${m.model}-${i}`} className="qa-param-row" onClick={() => onOpenModel(m)}>
                    {i === 0 && <td className="qa-param-model" rowSpan={m.rows.length}>{m.label}</td>}
                    <td>{r.parameter}</td>
                    <td>{r.candidates_or_rule}</td>
                    <td>{r.selected_or_fixed}</td>
                    <td>{r.selection_criterion}</td>
                  </tr>
                ))
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ParamHighlightCard({ model, onOpen }) {
  const finalRow = model.rows.find((r) => r.parameter_type === 'final_model') || model.rows[model.rows.length - 1];
  return (
    <div className="az-card qa-card qa-param-highlight" onClick={onOpen}>
      <div className="az-card-hd">
        <h3>{model.label}</h3>
        <p>{model.track}</p>
      </div>
      <div className="qa-detail-line"><b>핵심 파라미터</b> {model.rows.map((r) => r.parameter).join(', ')}</div>
      {model.detail && <div className="qa-detail-line"><b>구조</b> {model.detail.structure}</div>}
      <div className="qa-detail-line"><b>선정 기준</b> {finalRow?.selection_criterion}</div>
    </div>
  );
}

function ParamDetailModal({ model, onClose }) {
  useEffect(() => {
    function onKeyDown(e) { if (e.key === 'Escape') onClose(); }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  return (
    <div className="qa-modal-backdrop" onClick={onClose}>
      <div className="qa-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="qa-modal-hd">
          <h2>{model.label} 파라미터 상세</h2>
          <button type="button" className="qa-modal-close" onClick={onClose} aria-label="닫기">✕</button>
        </div>
        <div className="qa-modal-body">
          {model.detail && (
            <div className="qa-modal-section">
              <div className="qa-modal-section-hd">구조 요약</div>
              <div className="qa-detail-line"><b>입력</b> {model.detail.input}</div>
              <div className="qa-detail-line"><b>구조</b> {model.detail.structure}</div>
              <div className="qa-detail-line"><b>선정</b> {model.detail.selection}</div>
              <div className="qa-detail-line"><b>특징</b> {model.detail.characteristics}</div>
            </div>
          )}
          <div className="qa-modal-section">
            <div className="qa-modal-section-hd">전체 파라미터 ({model.track})</div>
            <div className="qa-modal-table-scroll">
            <table className="qa-param-table qa-param-table-full">
              <thead>
                <tr><th>구분</th><th>파라미터</th><th>탐색 범위/규칙</th><th>선택값/고정값</th><th>선정 단계</th><th>선정 기준</th></tr>
              </thead>
              <tbody>
                {model.rows.map((r, i) => (
                  <tr key={i}>
                    <td>{r.parameter_type}</td>
                    <td>{r.parameter}</td>
                    <td>{r.candidates_or_rule}</td>
                    <td>{r.selected_or_fixed}</td>
                    <td>{r.selection_stage}</td>
                    <td>{r.selection_criterion}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
