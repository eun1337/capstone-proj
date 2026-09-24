import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { api } from '../../api/client.js';
import EChart from '../../components/charts/EChart.jsx';
import ModelDetailModal from '../../components/ModelDetailModal.jsx';
import { STAT_MODEL_DETAIL } from './statModelDetailConfig.js';
import './analysis.css';
import './StatModelAnalysis.css';

const CENTERS = [
  { value: 'ALL', label: 'ALL' },
  { value: 'A', label: 'A' },
  { value: 'B', label: 'B' },
];
const METRICS = [
  { value: 'WAPE', label: 'WAPE' },
  { value: 'Bias', label: 'Bias' },
  { value: 'MAE', label: 'MAE' },
  { value: 'RMSE', label: 'RMSE' },
];
const BIAS_HORIZONS = [
  { value: 'h1', label: '1주 후' },
  { value: 'h2', label: '2주 후' },
  { value: 'h4', label: '4주 후' },
];
const CENTER_COMPARE_ORDER = ['A', 'B', 'ALL'];
const BIAS_SAFE_BAND = 20;
const EXCLUDED_MODELS = ['ARIMAX_S1', 'ARIMAX_S4'];

const PALETTE = ['#2563eb', '#10b981', '#8b5cf6', '#f59e0b', '#0ea5e9', '#ef4444', '#64748b'];

const metricLabel = (metric) => ['WAPE', 'Bias'].includes(metric) ? `${metric} (%)` : `${metric} (판매수량)`;
const metricGuide = (metric) => metric === 'Bias' ? '0에 가까울수록 편향이 작음' : '낮을수록 오차가 작음';
const LIGHT_TOOLTIP = { appendTo: 'body', confine: true, backgroundColor: '#fff', borderColor: '#cbd5e1', borderWidth: 1, textStyle: { color: '#334155' }, extraCssText: 'z-index:10000; box-shadow:0 6px 24px rgba(15,23,42,.12);' };
const INPUT_DATA = '센터 × SKU 주간 판매 이력 · A: 2021.01~2023.12, B: 2023.07~12';
const KEY_COLUMNS = [
  { col: 'center_id', meaning: '물류센터 식별자', purpose: '센터별 시계열 분리' },
  { col: 'sku_id', meaning: '판매관리 단위 식별자', purpose: '상품별 시계열 분리' },
  { col: 'week_st', meaning: '주 시작일', purpose: '시간순 정렬 및 외생변수 결합' },
];
function inputColumns(model, meta) {
  return [...KEY_COLUMNS, ...meta.columns.map((c) => c === QTY_COLUMN && !meta.seasonal
    ? { col: 'qty', meaning: '센터 × SKU 주간 판매수량', purpose: 'log1p 변환 후 판매 이력 학습' } : c)];
}

function fmtMetric(v) {
  if (!Number.isFinite(v)) return '-';
  const av = Math.abs(v);
  if (av >= 1000) return v.toExponential(2).replace('e+', 'e');
  return v.toFixed(2);
}

function fmtTooltip(v) {
  if (!Number.isFinite(v)) return '-';
  const av = Math.abs(v);
  if (av >= 1000) return v.toExponential(2).replace('e+', 'e');
  return v.toFixed(2);
}

function fmtAxisLabel(v) {
  if (Math.abs(v) >= 1000) return v.toExponential(1).replace('e+', 'e');
  return Number.isFinite(v) ? v.toFixed(1).replace(/\.0$/, '') : '-';
}

function axisTooltipFormatter(params) {
  const lines = params.map((p) => `${p.marker} ${p.seriesName}: ${fmtTooltip(p.value)}`);
  return `${params[0].axisValueLabel ?? params[0].axisValue}<br/>${lines.join('<br/>')}`;
}

const QTY_COLUMN = { col: 'qty_log1p', meaning: '센터 × SKU 주간 판매수량 (log1p 변환)', purpose: '과거 판매 흐름(추세·자기상관 패턴)을 모델에 반영' };

const ARIMA_PARAM = {
  range: '비계절 p ≤ 5, d ≤ 2, q ≤ 5 (차분 차수 d는 KPSS 검정으로 결정)',
  method: 'pmdarima auto_arima의 stepwise 탐색(전수탐색 아님)',
  criterion: 'AICc(정보기준) 최소 후보 선택',
  nonConvergence: 'AICc 최소 후보가 수렴하지 않으면, 같은 탐색에서 나온 후보 중 수렴한 후보만 모아 그중 AICc 최소 구조로 대체',
};

const MODEL_META = {
  ARIMA_S0: {
    label: 'ARIMA', oneLiner: '외생변수·계절성 없이 판매 이력만으로 학습하는 기준 모델',
    seasonal: false, exog: false,
    columns: [QTY_COLUMN],
    usedInfo: '판매이력만 사용 (계절패턴·외생변수 없음)',
  },
  SARIMA: {
    label: 'SARIMA', oneLiner: '반복되는 계절 패턴(명절 등)을 반영한 통계 대표모델',
    seasonal: true, exog: false,
    columns: [QTY_COLUMN],
    usedInfo: '판매이력 + 계절패턴 (외생변수 없음)',
  },
  ARIMAX_S1: {
    label: 'ARIMAX-S1', oneLiner: '경제지표(소비자심리·물가)를 추가한 ARIMA 실험 모델',
    seasonal: false, exog: true,
    columns: [
      QTY_COLUMN,
      { col: 'ccsi_lag_m1', meaning: '소비자심리지수(CCSI), 1개월 시차', purpose: '소비 심리 변화가 판매에 미치는 영향 반영' },
      { col: 'cpi_y1_prev', meaning: '소비자물가지수(CPI), 전년 기준값', purpose: '물가 수준 변화가 판매에 미치는 영향 반영' },
      { col: 'cpi_y2_prev_yoy', meaning: '소비자물가지수(CPI) 전전년 대비 YoY 변화율', purpose: '물가 변화율 추세를 반영' },
    ],
    usedInfo: '판매이력 + 경제지표 외생변수',
  },
  ARIMAX_S2: {
    label: 'ARIMAX-S2', oneLiner: 'COVID 시기 수요 변화를 추가한 ARIMA 실험 모델',
    seasonal: false, exog: true,
    columns: [
      QTY_COLUMN,
      { col: 'covid_flag', meaning: '코로나19 관련 기간 여부 플래그', purpose: '코로나 시기의 이례적 수요 변화를 반영' },
    ],
    usedInfo: '판매이력 + COVID 외생변수',
  },
  ARIMAX_S3: {
    label: 'ARIMAX-S3', oneLiner: '공휴일(설·추석) 전후 판매 변화를 추가한 ARIMA 실험 모델',
    seasonal: false, exog: true,
    columns: [
      QTY_COLUMN,
      { col: '공휴일_W0', meaning: '해당 주 자체에 공휴일(설·추석)이 포함', purpose: '명절이 있는 주의 판매 변화 반영' },
      { col: '공휴일_W-1', meaning: '해당 주의 1주 전이 공휴일 주', purpose: '명절 다음 주의 판매 변화 반영' },
      { col: '공휴일_W+1', meaning: '해당 주의 1주 후가 공휴일 주', purpose: '명절 이전 주의 판매 변화 반영' },
    ],
    usedInfo: '판매이력 + 공휴일 외생변수',
  },
  ARIMAX_S4: {
    label: 'ARIMAX-S4', oneLiner: '경제·COVID·공휴일 외생변수를 모두 결합한 ARIMA 실험 모델',
    seasonal: false, exog: true,
    columns: [
      QTY_COLUMN,
      { col: 'ccsi_lag_m1', meaning: '소비자심리지수(CCSI), 1개월 시차', purpose: '소비 심리 변화가 판매에 미치는 영향 반영' },
      { col: 'cpi_y1_prev', meaning: '소비자물가지수(CPI), 전년 기준값', purpose: '물가 수준 변화가 판매에 미치는 영향 반영' },
      { col: 'cpi_y2_prev_yoy', meaning: '소비자물가지수(CPI) 전전년 대비 YoY 변화율', purpose: '물가 변화율 추세를 반영' },
      { col: 'covid_flag', meaning: '코로나19 관련 기간 여부 플래그', purpose: '코로나 시기의 이례적 수요 변화를 반영' },
      { col: '공휴일_W0', meaning: '해당 주 자체에 공휴일(설·추석)이 포함', purpose: '명절이 있는 주의 판매 변화 반영' },
      { col: '공휴일_W-1', meaning: '해당 주의 1주 전이 공휴일 주', purpose: '명절 다음 주의 판매 변화 반영' },
      { col: '공휴일_W+1', meaning: '해당 주의 1주 후가 공휴일 주', purpose: '명절 이전 주의 판매 변화 반영' },
    ],
    usedInfo: '판매이력 + 경제·COVID·공휴일 외생변수 전체(7종)',
  },
  SARIMAX_S4: {
    label: 'SARIMAX', oneLiner: '계절 구조와 모든 외생변수를 함께 결합한 실험 모델',
    seasonal: true, exog: true,
    columns: [
      QTY_COLUMN,
      { col: 'ccsi_lag_m1', meaning: '소비자심리지수(CCSI), 1개월 시차', purpose: '소비 심리 변화가 판매에 미치는 영향 반영' },
      { col: 'cpi_y1_prev', meaning: '소비자물가지수(CPI), 전년 기준값', purpose: '물가 수준 변화가 판매에 미치는 영향 반영' },
      { col: 'cpi_y2_prev_yoy', meaning: '소비자물가지수(CPI) 전전년 대비 YoY 변화율', purpose: '물가 변화율 추세를 반영' },
      { col: 'covid_flag', meaning: '코로나19 관련 기간 여부 플래그', purpose: '코로나 시기의 이례적 수요 변화를 반영' },
      { col: '공휴일_W0', meaning: '해당 주 자체에 공휴일(설·추석)이 포함', purpose: '명절이 있는 주의 판매 변화 반영' },
      { col: '공휴일_W-1', meaning: '해당 주의 1주 전이 공휴일 주', purpose: '명절 다음 주의 판매 변화 반영' },
      { col: '공휴일_W+1', meaning: '해당 주의 1주 후가 공휴일 주', purpose: '명절 이전 주의 판매 변화 반영' },
    ],
    usedInfo: '판매이력 + 계절패턴 + 경제·COVID·공휴일 외생변수 전체(7종)',
  },
};

const DIVERGENCE_CAUSES = [
  { n: 1, title: '짧은 SKU별 Development', detail: '11주 / 14주 사례: 경제변수 계수를 안정적으로 추정하기에 짧은 이력' },
  { n: 2, title: '경제변수 간 강한 상관', detail: 'CCSI–CPI2 상관계수 −0.990 사례: 변수 간 높은 상관으로 계수 추정이 불안정' },
  { n: 3, title: '미래 경제변수의 학습범위 이탈', detail: 'CPI z-score 15.71 / 20.81 사례: 2024 경제변수가 Development 범위를 벗어남' },
  { n: 4, title: '불안정한 선형 외삽', detail: 'β × future exog가 log-scale 예측의 극단값을 생성' },
  { n: 5, title: '일부 AR/MA state 동학의 추가 증폭 가능', detail: '외생변수 외삽과 함께 일부 구조의 상태 동학이 극단 예측을 추가 증폭한 것으로 판단' },
  { n: 6, title: 'expm1 복원', detail: '작은 log-scale 차이가 원 수량으로 복원될 때 기하급수적으로 확대' },
];

export default function StatModelAnalysis() {
  const [center, setCenter] = useState('ALL');
  const [metric, setMetric] = useState('WAPE');
  const [selectedModel, setSelectedModel] = useState('SARIMA');
  const [detailOpen, setDetailOpen] = useState(false);

  const [centerCompare, setCenterCompare] = useState(null);
  const [compareError, setCompareError] = useState(null);

  const [holdoutPerf, setHoldoutPerf] = useState(null);
  useEffect(() => {
    let ignore = false;
    Promise.all(BIAS_HORIZONS.map(async ({ value }) => [value, await api.getQaStatVariableEffect({ center: 'ALL', horizon: value })]))
      .then((results) => { if (!ignore) setHoldoutPerf(Object.fromEntries(results)); })
      .catch(() => { if (!ignore) setHoldoutPerf(null); });
    return () => { ignore = true; };
  }, []);

  useEffect(() => {
    let ignore = false;
    setCompareError(null);
    setCenterCompare(null);
    Promise.all(CENTER_COMPARE_ORDER.map(async (value) => {
      const response = await api.getStatHeatmap({ center: value, metric });
      const row = response.rows.find((r) => r.model === selectedModel);
      return { center: value, isExtreme: row?.is_extreme === true, horizons: response.horizons, values: response.horizons.map((h) => row?.values[h] ?? null) };
    }))
      .then((series) => { if (!ignore) setCenterCompare({ model: selectedModel, label: MODEL_META[selectedModel].label, metric, horizons: series[0].horizons, series }); })
      .catch((e) => { if (!ignore) { setCompareError(e.message); setCenterCompare(null); } });
    return () => { ignore = true; };
  }, [selectedModel, metric]);

  return (
    <div className="sm-page">
      <div className="az-page-hd">
        <div>
          <h2>02 통계모델 분석</h2>
          <p>통계모델 중 어떤 구조가 가장 적합했는가</p>
        </div>
      </div>

      <div className="sm-dashboard-grid">
        <FamilyTreeCard selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <HeatmapCard center={center} onCenterChange={setCenter} pageMetric={metric} onMetricChange={setMetric} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <DetailCard selectedModel={selectedModel} holdoutPerf={holdoutPerf} onOpenDetail={() => setDetailOpen(true)} />
        <DivergenceCausesCard center={center} onCenterChange={setCenter} />
        <BiasMapCard center={center} onCenterChange={setCenter} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <CenterCompareCard data={centerCompare} error={compareError} metric={metric} selectedModel={selectedModel} />
      </div>

      {detailOpen && <DetailModal model={selectedModel} holdoutPerf={holdoutPerf} onClose={() => setDetailOpen(false)} />}
    </div>
  );
}

const FAMILY_NODES = [
  { id: 'ARIMA_S0', name: 'ARIMA', model: 'ARIMA_S0', x: 50, y: 8, hub: false },
  { id: 'ARIMAX_HUB', name: 'ARIMAX\n(외생변수)', model: null, x: 25, y: 34, hub: true },
  { id: 'SARIMA', name: 'SARIMA\n(계절성)', model: 'SARIMA', x: 76, y: 34, hub: false },
  { id: 'ARIMAX_S1', name: 'S1 경제', model: 'ARIMAX_S1', x: 10, y: 62, hub: false },
  { id: 'ARIMAX_S2', name: 'S2 COVID', model: 'ARIMAX_S2', x: 42, y: 62, hub: false },
  { id: 'ARIMAX_S3', name: 'S3 공휴일', model: 'ARIMAX_S3', x: 10, y: 90, hub: false },
  { id: 'ARIMAX_S4', name: 'S4 전체', model: 'ARIMAX_S4', x: 42, y: 90, hub: false },
  { id: 'SARIMAX_S4', name: 'SARIMAX', model: 'SARIMAX_S4', x: 76, y: 62, hub: false },
];
const FAMILY_EDGES = [
  { source: 'ARIMA_S0', target: 'ARIMAX_HUB', busY: 21 },
  { source: 'ARIMA_S0', target: 'SARIMA', busY: 21 },
  { source: 'ARIMAX_HUB', target: 'ARIMAX_S1', busY: 48 },
  { source: 'ARIMAX_HUB', target: 'ARIMAX_S2', busY: 48 },
  { source: 'ARIMAX_HUB', target: 'ARIMAX_S3', busY: 76 },
  { source: 'ARIMAX_HUB', target: 'ARIMAX_S4', busY: 76 },
  { source: 'SARIMA', target: 'SARIMAX_S4' },
];

function buildElbowLinks() {
  const nodeById = Object.fromEntries(FAMILY_NODES.map((n) => [n.id, n]));
  const waypoints = [];
  const links = [];
  FAMILY_EDGES.forEach(({ source, target, busY }) => {
    if (busY === undefined) {
      links.push({ source, target });
      return;
    }
    const s = nodeById[source];
    const t = nodeById[target];
    const a = `${source}__${target}__a`;
    const b = `${source}__${target}__b`;
    waypoints.push({ id: a, x: s.x, y: busY }, { id: b, x: t.x, y: busY });
    links.push({ source, target: a }, { source: a, target: b }, { source: b, target });
  });
  return { waypoints, links };
}

function buildFamilyTreeOption(selectedModel) {
  const nodes = FAMILY_NODES.map((n) => {
    const isSelected = n.model && n.model === selectedModel;
    return {
      id: n.id,
      name: n.name,
      model: n.model,
      value: [n.x, n.y],
      symbol: 'roundRect',
      symbolSize: n.hub ? [60, 30] : [56, 28],
      itemStyle: {
            color: isSelected ? '#2563eb' : '#eff6ff',
            borderColor: isSelected ? '#1d4ed8' : '#bfdbfe',
            borderWidth: isSelected ? 2 : 1,
          },
      label: {
        show: true,
        fontSize: 10,
        lineHeight: 12,
        color: isSelected ? '#fff' : '#1e40af',
        fontWeight: isSelected ? 700 : 600,
      },
    };
  });

  const { waypoints, links } = buildElbowLinks();
  const waypointNodes = waypoints.map((w) => ({ id: w.id, value: [w.x, w.y], symbol: 'none', symbolSize: 0 }));

  return {
    grid: { left: 28, right: 28, top: 20, bottom: 10 },
    xAxis: { type: 'value', min: 0, max: 100, show: false },
    yAxis: { type: 'value', min: 0, max: 100, show: false, inverse: true },
    tooltip: { show: false },
    series: [{
      type: 'graph',
      coordinateSystem: 'cartesian2d',
      roam: false,
      symbol: 'roundRect',
      data: [...nodes, ...waypointNodes],
      links,
      edgeSymbol: ['none', 'none'],
      lineStyle: { color: '#cbd5e1', width: 1.3, curveness: 0 },
    }],
  };
}

const TREE_TIP_NODES = FAMILY_NODES.filter((n) => n.model);
const TREE_GRID = { left: 28, right: 28, top: 20, bottom: 10, height: 300 };

function treeNodeBtnStyle(n) {
  return {
    left: `calc(${TREE_GRID.left}px + (100% - ${TREE_GRID.left + TREE_GRID.right}px) * ${n.x / 100} - 28px)`,
    top: `calc(${TREE_GRID.top}px + (100% - ${TREE_GRID.top + TREE_GRID.bottom}px) * ${n.y / 100} - 14px)`,
  };
}

function TreeTooltip({ anchor, model, meta }) {
  const ref = useRef(null);
  const [position, setPosition] = useState({ left: 8, top: 8, visibility: 'hidden' });
  useLayoutEffect(() => {
    function place() {
      const rect = anchor.getBoundingClientRect();
      const tip = ref.current.getBoundingClientRect();
      const margin = 8;
      let left = rect.right + margin;
      if (left + tip.width > window.innerWidth - margin) left = rect.left - tip.width - margin;
      let top = rect.top;
      if (top + tip.height > window.innerHeight - margin) top = rect.bottom - tip.height;
      setPosition({ left: Math.max(margin, Math.min(left, window.innerWidth - tip.width - margin)), top: Math.max(margin, Math.min(top, window.innerHeight - tip.height - margin)), visibility: 'visible' });
    }
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => { window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); };
  }, [anchor, model]);
  return createPortal(<div ref={ref} id="sm-model-tooltip" role="tooltip" className="sm-tree-tip" style={position}>
    <div className="sm-tree-tip-title">{meta.label}</div>
    <dl className="sm-tree-tip-content">
      <div><dt>모델 설명</dt><dd>{meta.oneLiner}</dd></div>
      <div><dt>사용 데이터</dt><dd>{INPUT_DATA}</dd></div>
      <div><dt>사용 컬럼</dt><dd>{inputColumns(model, meta).map((c) => c.col).join(' · ')}</dd></div>
      <div><dt>구조</dt><dd>계절성 {meta.seasonal ? '사용' : '없음'} · 외생변수 {meta.exog ? '사용' : '없음'}</dd></div>
      <div><dt>활용 목적</dt><dd>{meta.usedInfo}에 따른 예측 성능 비교</dd></div>
    </dl>
  </div>, document.body);
}

function FamilyTreeCard({ selectedModel, onSelectModel }) {
  const [hoverId, setHoverId] = useState(null);
  const [anchor, setAnchor] = useState(null);
  const hoverNode = TREE_TIP_NODES.find((n) => n.id === hoverId);
  const hoverMeta = hoverNode && MODEL_META[hoverNode.model];

  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>1. 통계모델 확장 구조</h3>
        <p>ARIMA → ARIMAX / SARIMA → SARIMAX · 발산 모델은 별도 제외</p>
      </div>
      <div className="sm-tree-wrap">
        <EChart
          fill
          option={buildFamilyTreeOption(selectedModel)}
          onEvents={{
            click: (p) => { if (p.data?.model) onSelectModel(p.data.model); },
          }}
        />
        <div className="sm-tree-overlay">
          {TREE_TIP_NODES.map((n) => (
            <button
              key={n.id}
              type="button"
              className="sm-tree-node-btn"
              style={treeNodeBtnStyle(n)}
              onMouseEnter={(e) => { setAnchor(e.currentTarget); setHoverId(n.id); }}
              onMouseLeave={() => setHoverId((cur) => (cur === n.id ? null : cur))}
              onFocus={(e) => { setAnchor(e.currentTarget); setHoverId(n.id); }}
              onBlur={() => setHoverId((cur) => (cur === n.id ? null : cur))}
              onClick={() => onSelectModel(n.model)}
              aria-describedby={hoverId === n.id ? "sm-model-tooltip" : undefined}
              aria-label={`${MODEL_META[n.model]?.label || n.model} 상세 정보`}
            />
          ))}
        </div>
        {hoverNode && hoverMeta && anchor && <TreeTooltip anchor={anchor} model={hoverNode.model} meta={hoverMeta} />}
      </div>
      <p className="sm-family-note" title={`${ARIMA_PARAM.range} · ${ARIMA_PARAM.method} · ${ARIMA_PARAM.criterion}`}>센터 × SKU × 주 단위 · AICc 기준 구조 결정 · 1주·2주·4주 후 평가 ⓘ</p>
    </div>
  );
}

function MetricMenu({ value, onChange }) {
  return <select className="sm-metric-menu" aria-label="평가지표 선택" value={value} onChange={(event) => onChange(event.target.value)}>{METRICS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}</select>;
}

function CardFilters({ center, onCenterChange, metric, onMetricChange, children }) {
  return <div className="sm-card-filters"><select className="sm-metric-menu" aria-label="센터 선택" title={center === 'ALL' ? 'A/B센터 전체 예측행을 합산해 계산한 성능' : `${center}센터 예측행 기준 성능`} value={center} onChange={(event) => onCenterChange(event.target.value)}>{CENTERS.map((item) => <option key={item.value} value={item.value}>{item.label === 'ALL' ? '전체' : `${item.label}센터`}</option>)}</select>{metric && <MetricMenu value={metric} onChange={onMetricChange} />}{children}</div>;
}

function HeatmapCard({ center, onCenterChange, pageMetric: metric, onMetricChange: setMetric, selectedModel, onSelectModel }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;
    setError(null);
    setData(null);
    api.getStatCommonHeatmap({ center, metric })
      .then((d) => { if (!ignore) setData(d); })
      .catch((e) => { if (!ignore) { setError(e.message); setData(null); } });
    return () => { ignore = true; };
  }, [center, metric]);

  return (
    <div className="az-card">
      <div className="az-card-hd sm-card-hd-row">
        <div>
          <h3>2. 동일 조건 모델 성능 비교</h3>
          <p>공통 예측 row 기준 · {data?.metric || metric} · {metricGuide(data?.metric || metric)}</p>
        </div>
        <CardFilters center={center} onCenterChange={onCenterChange} metric={metric} onMetricChange={setMetric} />
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <>
          <div className="sm-equal-heatmap-slot"><EChart
            height={190}
            option={buildHeatmapOption(data, selectedModel)}
            onEvents={{
              click: (p) => { if (p.data?.model) onSelectModel(p.data.model); },
            }}
          />
          </div>
          <div className="sm-scale-bar sm-heatmap-legend">
            <span>{data.metric === 'Bias' ? '0에 가까움(우수)' : '낮음(우수)'}</span>
            <span className="sm-scale-gradient" />
            <span>{data.metric === 'Bias' ? '|Bias| 큼(미흡)' : '높음(미흡)'}</span><span className="sm-legend-note">{data.metric === 'Bias' ? '굵은 값: 시점별 최소 |Bias|' : '굵은 값: 시점별 최저 오차'}</span>
          </div>
        </>
      )}
    </div>
  );
}

function buildHeatmapOption(data, selectedModel) {
  data = { ...data, rows: data.rows.filter((r) => !r.is_extreme) };
  const horizons = data.horizons;
  const models = data.rows.map((r) => r.label);
  const magnitudes = data.rows.flatMap((r) => Object.values(r.values)).filter(Number.isFinite).map((v) => data.metric === 'Bias' ? Math.abs(v) : v);
  const min = magnitudes.length ? Math.min(...magnitudes) : 0;
  const max = magnitudes.length ? Math.max(...magnitudes) : 1;
  const selectedIdx = data.rows.findIndex((r) => r.model === selectedModel);

  const cells = [];
  data.rows.forEach((row, yi) => {
    horizons.forEach((h, xi) => {
      const raw = row.values[h];
      if (!Number.isFinite(raw)) return;
      const clipped = Math.min(Math.max(data.metric === 'Bias' ? Math.abs(raw) : raw, min), max);
      const best = Math.min(...data.rows.map((candidate) => candidate.values[h]).filter(Number.isFinite).map((v) => data.metric === 'Bias' ? Math.abs(v) : v));
      const isBest = clipped === best;
      cells.push({ value: [xi, yi, clipped], raw, model: row.model,
        label: { fontWeight: isBest ? 800 : 500, fontSize: isBest ? 13 : 12 },
        modelLabel: row.label, nRows: row.n_rows?.[h],
      });
    });
  });

  return {
    grid: { left: 82, right: 82, top: 10, height: 130 },
    tooltip: {
      ...LIGHT_TOOLTIP,
      formatter: (p) => `${p.data.modelLabel} · ${BIAS_HORIZONS.find((item) => item.value === horizons[p.data.value[0]])?.label || horizons[p.data.value[0]]}<br/>${data.metric}: ${fmtTooltip(p.data.raw)}`
        + (p.data.nRows ? `<br/>7개 모델 전체 공통 panel row ${p.data.nRows.toLocaleString()}건` : ''),
    },
    xAxis: {
      type: 'category', data: horizons.map((h) => BIAS_HORIZONS.find((item) => item.value === h)?.label || h),
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11 },
    },
    yAxis: {
      type: 'category', data: models, inverse: true,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11 },
    },
    visualMap: { min, max, show: false, seriesIndex: 0, inRange: { color: ['#7cb2ef', '#a0c8f4', '#c3dcf8', '#e0edfc', '#f5f9ff'] } },
    series: [{
      type: 'heatmap',
      data: cells,
      label: { show: true, formatter: (p) => fmtMetric(p.data.raw), fontSize: 12, color: '#172554' },
      emphasis: { itemStyle: { borderWidth: 0 } },
    }, {
      type: 'custom', silent: true, z: 20, clip: false,
      data: selectedIdx >= 0 ? [[0, selectedIdx]] : [],
      renderItem: (params, chartApi) => {
        const first = chartApi.coord([0, selectedIdx]);
        const last = chartApi.coord([horizons.length - 1, selectedIdx]);
        const [width, height] = chartApi.size([1, 1]).map(Math.abs);
        return { type: 'rect', shape: { x: first[0] - width / 2 + 1, y: first[1] - height / 2 + 1, width: last[0] - first[0] + width - 2, height: height - 2 }, style: { fill: 'transparent', stroke: '#2563eb', lineWidth: 2 } };
      },
    }],
  };
}

function BiasMapCard({ center, onCenterChange, selectedModel, onSelectModel }) {
  const [horizon, setHorizon] = useState('h1');
  const [profiles, setProfiles] = useState(null);
  const data = profiles?.[horizon];
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;
    setError(null);
    setProfiles(null);
    Promise.all(BIAS_HORIZONS.map(async ({ value: horizonValue }) => {
      const response = await api.getStatCommonWapeBias({ center, horizon: horizonValue });
      return [horizonValue, { points: response.points }];
    }))
      .then((results) => { if (!ignore) setProfiles(Object.fromEntries(results)); })
      .catch((e) => { if (!ignore) { setError(e.message); setProfiles(null); } });
    return () => { ignore = true; };
  }, [center]);

  return (
    <div className="az-card">
      <div className="az-card-hd sm-card-hd-row">
        <div>
          <h3>5. 오차 크기와 과대·과소예측 안정성</h3>
          <p>WAPE ↓ 오차 작을수록 우수 · Bias → 0에 가까울수록 우수</p>
        </div>
        <CardFilters center={center} onCenterChange={onCenterChange}><select className="sm-metric-menu" aria-label="예측시점 선택" value={horizon} onChange={(event) => setHorizon(event.target.value)}>{BIAS_HORIZONS.map((h) => <option key={h.value} value={h.value}>{h.label}</option>)}</select></CardFilters>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <>
          <EChart
            fill
            option={buildBiasOption(data, selectedModel, center, horizon, profiles)}
            onEvents={{
              click: (p) => { if (p.data?.model) onSelectModel(p.data.model); },
            }}
          />
          <div className="sm-bias-legend">
            <span><i className="sm-dot" style={{ background: '#2563eb' }} />정상 후보</span>
            <span><i className="sm-dot" style={{ background: '#047857' }} />선정 모델(SARIMA)</span>

          </div>
        </>
      )}
    </div>
  );
}

function buildBiasOption(data, selectedModel, center, horizon, profiles) {
  const points = data.points.filter((p) => Number.isFinite(p.wape) && Number.isFinite(p.bias));
  const allPoints = Object.values(profiles).flatMap((profile) => profile.points).filter((p) => Number.isFinite(p.wape) && Number.isFinite(p.bias));
  const biasMin = Math.min(...allPoints.map((p) => p.bias), -BIAS_SAFE_BAND);
  const biasMax = Math.max(...allPoints.map((p) => p.bias), BIAS_SAFE_BAND);
  const biasPadding = Math.max((biasMax - biasMin) * .06, 1);
  const wapeMin = allPoints.length ? Math.min(...allPoints.map((p) => p.wape)) : 0;
  const wapeMax = allPoints.length ? Math.max(...allPoints.map((p) => p.wape)) : 1;
  const wapePadding = Math.max((wapeMax - wapeMin) * .12, 1);
  const yMin = Math.max(0, wapeMin - wapePadding);
  const yMax = wapeMax + wapePadding;
  const tickStep = (span) => {
    const rough = Math.max(span / 5, .1);
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    return [1, 2, 5, 10].find((step) => step * magnitude >= rough) * magnitude;
  };
  const interval = tickStep(Math.max(biasMax - biasMin + 2 * biasPadding, yMax - yMin));
  const xLow = Math.floor((biasMin - biasPadding) / interval) * interval;
  const xHigh = Math.ceil((biasMax + biasPadding) / interval) * interval;
  const yLow = Math.max(0, Math.floor(yMin / interval) * interval);
  const yHigh = Math.ceil(yMax / interval) * interval;
  const reference = allPoints.filter((point) => point.model === 'SARIMA');
  const referenceBias = Math.max(...reference.map((point) => Math.abs(point.bias)), BIAS_SAFE_BAND) + interval * .5;
  const referenceWape = Math.max(...reference.map((point) => point.wape), yLow) + interval * .5;
  return {
    grid: { left: 52, right: 32, top: 28, bottom: 35 },
    tooltip: { ...LIGHT_TOOLTIP, formatter: (p) => `${p.data.modelLabel}<br/>센터: ${center === 'ALL' ? '전체' : center} · 예측시점: ${BIAS_HORIZONS.find((item) => item.value === horizon)?.label || horizon}<br/>WAPE: ${fmtTooltip(p.data.wape)}%<br/>Bias: ${fmtTooltip(p.data.bias)}%${p.data.nRows ? `<br/>7개 모델 전체 공통 panel row ${p.data.nRows.toLocaleString()}건` : ''}` },
    xAxis: { type: 'value', name: 'Bias (%)', nameLocation: 'middle', nameGap: 22, min: xLow, max: xHigh, interval, axisLabel: { formatter: fmtAxisLabel }, splitLine: { lineStyle: { color: '#eef2f6' } } },
    yAxis: { type: 'value', name: 'WAPE (%) · 큼 ↑', min: yLow, max: yHigh, interval, axisLabel: { formatter: fmtAxisLabel }, splitLine: { lineStyle: { color: '#eef2f6' } } },
    series: [{ type: 'scatter',
      markArea: { silent: true,
        itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(16,185,129,.03)' }, { offset: 1, color: 'rgba(16,185,129,.23)' }] } },
        label: { color: '#047857', fontSize: 10, position: 'insideBottom', formatter: '낮은 오차 + 적은 편향' },
        data: reference.length ? [[{ xAxis: Math.max(xLow, -referenceBias), yAxis: yLow }, { xAxis: Math.min(xHigh, referenceBias), yAxis: Math.min(yHigh, referenceWape) }]] : [],
      },
      data: points.map((p) => {
        const color = p.model === 'SARIMA' ? '#059669' : '#2563eb';
        return { ...p, symbol: 'circle', modelLabel: p.label, nRows: p.n_rows, value: [p.bias, p.wape],
          symbolSize: 14,
          itemStyle: { color, borderColor: color, borderWidth: 1, opacity: 1 },
          label: { show: true, formatter: p.label, position: 'top', distance: 12, fontSize: 11, fontWeight: 650, color, backgroundColor: 'rgba(255,255,255,.9)', padding: [2, 3] },
          emphasis: { scale: false },
        };
      }),
      labelLayout: (params) => ({
        x: params.rect.x + params.rect.width / 2 + (params.dataIndex % 2 ? 24 : -24),
        y: params.rect.y - 18 - Math.floor(params.dataIndex / 2) * 18,
        align: params.dataIndex % 2 ? 'left' : 'right',
        verticalAlign: 'bottom', hideOverlap: false, moveOverlap: 'shiftY',
      }),
      labelLine: { show: true, length2: 8, lineStyle: { color: '#94a3b8', width: 1 } },
    }],
  };
}

function needsCenterMatrix(data) {
  const magnitudes = data.series.flatMap((series) => series.values)
    .filter(Number.isFinite).map(Math.abs).filter((value) => value > 0);
  if (!magnitudes.length) return false;
  const largest = Math.max(...magnitudes);
  return largest >= 1e6 || largest / Math.min(...magnitudes) >= 1e3;
}

function CenterResultMatrix({ data }) {
  const classifiedExtreme = data.series.some((series) => series.isExtreme);
  return (
    <div className="sm-center-matrix">
      <div className="sm-center-matrix-note">
        <span>극단적인 값으로 일반 축에서 비교하기 어려운 결과입니다</span>
        {classifiedExtreme && <span className="sm-divergence-badge" title="기존 API의 모델 단위 WAPE 발산 분류">발산</span>}
      </div>
      <table className="sm-center-result-table">
        <caption>{metricLabel(data.metric)}</caption>
        <thead><tr><th scope="col">센터</th>{BIAS_HORIZONS.map((h) => <th scope="col" key={h.value}>{h.label}</th>)}</tr></thead>
        <tbody>{CENTER_COMPARE_ORDER.map((center) => {
          const series = data.series.find((item) => item.center === center);
          return <tr key={center}>
            <th scope="row">{center === 'ALL' ? '전체' : `${center}센터`}</th>
            {BIAS_HORIZONS.map((h) => {
              const index = data.horizons.indexOf(h.value);
              const value = index >= 0 ? series?.values[index] : null;
              return <td key={h.value}>{Number.isFinite(value) ? fmtTooltip(value) : '데이터 없음'}</td>;
            })}
          </tr>;
        })}</tbody>
      </table>
    </div>
  );
}

function CenterCompareCard({ data, error, metric, selectedModel }) {
  const excluded = EXCLUDED_MODELS.includes(selectedModel);

  return (
    <div className="az-card">
      <div className="az-card-hd sm-card-hd-row">
        <div>
          <h3>6. {selectedModel === 'SARIMA' ? '선정 SARIMA의 센터별 성능' : `${MODEL_META[selectedModel].label}의 센터별 성능`}</h3>
          <p>{metricLabel(metric)} · {metricGuide(metric)} <span title="ALL은 A/B 단순 평균이 아닌 전체 관측치를 합산한 pooled metric입니다.">ⓘ</span></p>
        </div>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {excluded && <div className="sm-excluded-notice">2024 Holdout 안정성 실패로 센터별 정상 비교에서 제외된 모델입니다. 발산 결과 카드를 확인하세요.</div>}
      {!excluded && !error && data && (data.series.some((s) => s.values.some(Number.isFinite))
        ? (needsCenterMatrix(data) ? <CenterResultMatrix data={data} /> : <EChart fill option={buildCenterCompareOption(data)} />)
        : <div className="az-hint">해당 지표 데이터 없음</div>)}
      {!excluded && selectedModel === 'SARIMA' && <p className="sm-center-stability">A/B센터 모두 1주·2주·4주 후 성능 급변 없이 유지</p>}
    </div>
  );
}

function buildCenterCompareOption(data) {
  const visibleSeries = data.series.filter((s) => s.center !== 'ALL');
  const categories = visibleSeries.map((s) => `${s.center}센터`);
  return {
    grid: { left: 62, right: 42, top: 34, bottom: 38 },
    tooltip: { ...LIGHT_TOOLTIP, trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (params) => `${metricLabel(data.metric)}<br/>${axisTooltipFormatter(params)}` },
    color: ['#1d4ed8', '#3b82f6', '#93c5fd'],
    legend: { bottom: 0, textStyle: { color: '#475569', fontSize: 11 } },
    xAxis: { type: 'value', name: metricLabel(data.metric), splitLine: { lineStyle: { color: '#f1f5f9' } }, axisLabel: { color: '#64748b', formatter: fmtAxisLabel } },
    yAxis: { type: 'category', inverse: true, data: categories, axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b' } },
    series: data.horizons.map((horizon, horizonIdx) => ({
      name: BIAS_HORIZONS.find((item) => item.value === horizon)?.label || horizon,
      type: 'bar',
      barMaxWidth: 25,
      label: { show: true, position: 'right', fontSize: 10, distance: 5, fontWeight: 700, color: '#1e293b', formatter: (p) => fmtTooltip(p.value) },
      data: visibleSeries.map((s) => s.values[horizonIdx]),
    })),
  };
}

function buildExtremeHeatmap(rows, horizons, center) {
  const cells = rows.flatMap((row, y) => horizons.flatMap((h, x) => {
    const raw = row.values[h];
    return Number.isFinite(raw) ? [{ value: [x, y, Math.log10(Math.max(Math.abs(raw), 1))], raw, label: row.label, horizon: h }] : [];
  }));
  return {
    grid: { left: 82, right: 82, top: 10, height: 52 },
    tooltip: { ...LIGHT_TOOLTIP, formatter: (p) => `${p.data.label} · ${center === 'ALL' ? '전체' : `${center}센터`} · ${BIAS_HORIZONS.find((item) => item.value === p.data.horizon)?.label || p.data.horizon}<br/>WAPE: ${fmtTooltip(p.data.raw)}` },
    xAxis: { type: 'category', data: horizons.map((h) => BIAS_HORIZONS.find((item) => item.value === h)?.label || h), axisLine: { show: false }, axisTick: { show: false } },
    yAxis: { type: 'category', inverse: true, data: rows.map((r) => r.label), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { fontSize: 10 } },
    visualMap: { show: false, min: Math.min(...cells.map((c) => c.value[2]), 0), max: Math.max(...cells.map((c) => c.value[2]), 1), inRange: { color: ['#fff5f5', '#fee2e2', '#fecaca', '#f8a6a6', '#ef8585'] } },
    series: [{ type: 'heatmap', data: cells, label: { show: true, formatter: (p) => `${fmtTooltip(p.data.raw)}`, fontSize: 12, fontWeight: 650, color: '#641b1b' }, itemStyle: { borderColor: '#fff', borderWidth: 2 } }],
  };
}

function CoverageMiniTable({ entries }) {
  return <table className="sm-coverage-table">
    <thead><tr><th>모델</th><th>정상 예측</th><th>constant</th><th>naive_mean</th><th>cold-start</th></tr></thead>
    <tbody>{entries.map((e) => <tr key={e.model}>
      <td>{e.label}</td>
      <td>{(e.normal_rate * 100).toFixed(1)}% ({e.n_normal.toLocaleString()}건)</td>
      <td>{e.n_fallback_constant.toLocaleString()}</td>
      <td>{e.n_fallback_naive_mean.toLocaleString()}</td>
      <td>{e.n_has_observed_history_false.toLocaleString()}</td>
    </tr>)}</tbody>
  </table>;
}

function DivergenceCausesCard({ center, onCenterChange }) {
  const [heatmap, setHeatmap] = useState(null);
  const [coverage, setCoverage] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    let ignore = false;
    setHeatmap(null);
    setError(null);
    api.getStatCommonHeatmap({ center, metric: 'WAPE' })
      .then((result) => { if (!ignore) setHeatmap(result); })
      .catch((err) => { if (!ignore) setError(err.message); });
    return () => { ignore = true; };
  }, [center]);
  useEffect(() => {
    let ignore = false;
    api.getStatCoverage().then((result) => { if (!ignore) setCoverage(result.entries); }).catch(() => {});
    return () => { ignore = true; };
  }, []);
  useEffect(() => {
    if (!open) return;
    const close = (event) => { if (event.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [open]);

  const extremeRows = heatmap ? heatmap.rows.filter((r) => r.is_extreme) : null;
  const extremeCoverage = coverage ? coverage.filter((e) => e.is_extreme) : null;

  return <div className="az-card sm-divergence-card sm-clickable-card" role="button" tabIndex={0} aria-label="발산 결과 상세 진단 보기" aria-haspopup="dialog"
    onClick={(event) => { if (!event.target.closest('select') && !open) setOpen(true); }}
    onKeyDown={(event) => { if (event.target === event.currentTarget && ['Enter', ' '].includes(event.key)) { event.preventDefault(); setOpen(true); } }}>
    <div className="az-card-hd sm-card-hd-row"><div><h3>4. ARIMAX-S1·S4 수치 발산 → 비교 제외</h3><p>정상적인 오차 범위를 넘어 예측값이 폭증함 · 공통 row 기준 ({center === 'ALL' ? '전체' : `${center}센터`})</p></div><CardFilters center={center} onCenterChange={onCenterChange} /></div>
    {error && <div className="az-hint">{error}</div>}
    {!error && !extremeRows && <div className="az-hint">불러오는 중...</div>}
    {extremeRows && extremeRows.length > 0 && <div className="sm-equal-heatmap-slot"><EChart height={88} option={buildExtremeHeatmap(extremeRows, heatmap.horizons, center)} /></div>}
    <div className="sm-scale-bar sm-heatmap-legend sm-extreme-legend"><span>작음</span><span className="sm-scale-gradient sm-extreme-gradient" /><span>큼</span></div>
    {open && createPortal(<div className="sm-modal-backdrop" onClick={(event) => { event.stopPropagation(); setOpen(false); }}><div className="sm-modal" role="dialog" aria-modal="true" aria-labelledby="sm-diagnostic-title" onClick={(e) => e.stopPropagation()}><div className="sm-modal-hd"><h3 id="sm-diagnostic-title">발산 결과 상세 진단</h3><button autoFocus className="sm-modal-close" aria-label="닫기" onClick={() => setOpen(false)}>✕</button></div><div className="sm-modal-body">
      {DIVERGENCE_CAUSES.map((cause) => <section className="sm-modal-section" key={cause.n}><h4>{cause.n}. {cause.title}</h4><p>{cause.detail}</p></section>)}
      {extremeCoverage && <section className="sm-modal-section"><h4>7. 모델별 정상/fallback 현황(own 전체 population 기준)</h4><CoverageMiniTable entries={extremeCoverage} /><p className="sm-modal-note">정상 예측 비율이 낮을수록(즉 constant/naive_mean/cold-start로 대체된 행이 많을수록) auto_arima/SARIMAX fit 자체가 실패했다는 뜻입니다. 위 WAPE 극단값은 이 fallback 비율과는 별개로, "정상"으로 분류된 fit 결과 자체가 발산한 경우도 포함합니다.</p></section>}
      <p className="sm-modal-verdict">현재 검증 범위에서는 정렬/저장 오류가 아니라 모델의 예측 안정성 실패로 판단</p>
    </div></div></div>, document.body)}
  </div>;
}

function statPerfEntry(holdoutPerf, horizon, model) {
  return holdoutPerf?.[horizon]?.entries.find((e) => e.key === model) || null;
}

function statStatus(model) {
  if (model === 'SARIMA') return { tone: 'final', text: '최종 선정' };
  if (EXCLUDED_MODELS.includes(model)) return { tone: 'excluded', text: '발산으로 비교 제외' };
  return { tone: 'normal', text: '미선정' };
}

function modelUiSummary(model, meta) {
  const exog = meta.columns.filter((item) => item.col !== 'qty_log1p').map((item) => item.col);
  const isArimax = model.startsWith('ARIMAX');
  const isSarimax = model.startsWith('SARIMAX');
  const description = model === 'ARIMAX_S2'
    ? 'ARIMA에 COVID 시기 여부를 추가하여 수요 변화 반영 여부를 검증한 모델'
    : meta.oneLiner;
  const structure = model === 'ARIMA_S0' ? 'ARIMA (p,d,q)'
    : model === 'SARIMA' ? 'ARIMA (p,d,q) + 계절구조 (P,D,Q,m)'
      : isSarimax ? '계절구조 (p,d,q)(P,D,Q,m) + 외부변수' : 'ARIMA (p,d,q) + 외부변수';
  const decision = model === 'ARIMA_S0' ? ['상품별 AICc 최솟값 선택', 'auto_arima 탐색으로 비계절 구조 결정']
    : model === 'SARIMA' ? ['상품별 AICc 최솟값 선택', 'ARIMA 구조를 유지하고 계절구조를 추가 탐색']
      : isSarimax ? ['SARIMA와 동일 후보에서 독립 적합', '외부변수를 포함한 상태에서 구조별 수렴·AICc 확인']
        : ['ARIMA 확정 구조 그대로 사용', '(p,d,q)는 재탐색하지 않고 외부변수만 추가'];
  return {
    description, exog,
    input: ['qty_log1p', ...exog].join(' + '),
    inputNote: exog.length ? `주간 판매량 시계열에 ${exog.join(' · ')} 추가` : '주간 판매량 시계열만 사용',
    structure,
    structureNote: model === 'ARIMA_S0' ? '비계절 시계열 기준 구조' : model === 'SARIMA' ? '비계절 구조에 반복 계절 패턴 결합' : isSarimax ? '계절 시계열 구조에 외부정보 결합' : '비계절 시계열 구조에 외부정보 결합',
    decision,
    baseStructure: model === 'ARIMA_S0' ? '상품별 (p,d,q)를 새로 탐색' : model === 'SARIMA' ? 'ARIMA 단계에서 확정한 상품별 (p,d,q) 구조 사용' : isSarimax ? 'SARIMA와 동일한 상품별 후보 구조 사용' : 'ARIMA 단계에서 확정한 상품별 (p,d,q) 구조 사용',
  };
}

function variableRole(column) {
  if (column === 'qty_log1p') return '예측 대상 시계열';
  if (column === 'covid_flag') return 'COVID 시기의 수요 변화를 설명하는 외부변수';
  if (column.startsWith('ccsi')) return '소비심리 변화가 수요에 미치는 영향을 설명하는 외부변수';
  if (column.startsWith('cpi')) return '물가 수준·변화가 수요에 미치는 영향을 설명하는 외부변수';
  if (column.startsWith('공휴일_')) return '명절 전후 수요 변화를 설명하는 외부변수';
  return '모델 입력 변수';
}

function detailPreprocessing(preprocessing) {
  return preprocessing.flatMap((item) => {
    if (item.target === 'target(qty)') return [{ target: '판매수량', method: 'log(판매수량 + 1) 변환', reason: '0 판매량을 포함해 로그 변환하고 큰 판매량 값의 영향을 완화' }];
    if (item.target === '예측값') return [{ target: '예측값', method: 'expm1 역변환 후 0 미만은 0으로 보정', reason: '원래 판매수량 단위로 복원' }];
    if (!item.target.includes('미수렴')) return [{ target: item.target.replace('target(qty)', '판매수량'), method: item.method, reason: item.reason }];
    return [
      { target: '미수렴', method: '같은 탐색에서 수렴한 후보 중 AICc가 가장 작은 구조 사용', reason: '모델 후보 일부가 적합되지 않는 경우 대응' },
      { target: '이력 부족 / cold-start', method: 'KAN 소분류 → 중분류 → 대분류 → 센터 평균 순으로 보완', reason: '자체 판매이력이 부족한 상품의 예측값 보완' },
      { target: '최종 fallback', method: 'naive_mean으로 과거 이력 평균 사용', reason: '앞선 방식 적용이 어려운 경우 예측값 생성' },
    ];
  });
}

function detailParameters(model, config, summary) {
  if (model.startsWith('ARIMAX')) return [
    ['(p,d,q)', '비계절 AR·차분·MA 구조', 'ARIMA 단계에서 상품별로 확정한 값 그대로 사용, 재탐색 없음'],
    ['with_intercept', '상수항 포함 여부', 'ARIMA 단계에서 확정한 상품별 설정 그대로 사용'],
    ['외부변수', '추가 설명변수', summary.exog.join(' · ')],
    ['구조 결정', 'ARIMAX 구조 적용 방식', `기존 ARIMA 구조는 유지하고 ${summary.exog.join(' · ')}만 추가`],
  ];
  if (model === 'SARIMAX_S4') return [
    ['(p,d,q)', '비계절 AR·차분·MA 구조', 'SARIMA와 동일한 상품별 후보 구조 사용'],
    ['(P,D,Q,m)', '계절 구조', 'SARIMA와 동일한 후보 범위에서 외부변수 포함 상태로 독립 적합'],
    ['외부변수', '추가 설명변수', summary.exog.join(' · ')],
    ['구조 결정', 'SARIMAX 구조 적용 방식', '외부변수를 포함해 수렴 여부와 AICc를 독립적으로 확인'],
  ];
  return config.actualParams.map((item) => [item.param, item.purpose, item.value]);
}

function fmtDetailMetric(value) {
  if (!Number.isFinite(value)) return '-';
  if (Math.abs(value) >= 1e12) return value.toExponential(2).replace('e+', 'e');
  return value.toLocaleString('ko-KR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function DetailCard({ selectedModel, holdoutPerf, onOpenDetail }) {
  const meta = MODEL_META[selectedModel];
  const config = STAT_MODEL_DETAIL[selectedModel];
  if (!meta || !config) return null;

  const status = statStatus(selectedModel);
  const summary = modelUiSummary(selectedModel, meta);

  return (
    <div className="az-card sm-detail-card-wrap sm-clickable-summary" role="button" tabIndex={0} aria-haspopup="dialog" onClick={onOpenDetail} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onOpenDetail(); } }}>
      <div className="az-card-hd sm-card-hd-row">
        <div><h3>{selectedModel === 'SARIMA' ? '3. 통계 대표모델 SARIMA' : `3. 선택 모델 ${meta.label}`}</h3></div>
      </div>
      <div className={`az-detail-title sm-model-status is-${status.tone}`}><strong>{meta.label}</strong><span>{selectedModel === 'SARIMA' ? '통계 대표모델' : status.text}</span></div>
      <p className="sm-model-summary-line">{selectedModel === 'SARIMA' ? '상품별 과거 판매량에서 반복되는 계절 패턴까지 반영하는 시계열 모델' : summary.description}</p>
      <div className="sm-sarima-structure sm-sarima-main-boxes">
        <div><b>입력 변수</b><span><code>{summary.input}</code></span><small>{selectedModel === 'SARIMA' ? '주간 판매량을 log(판매수량+1)로 변환' : summary.inputNote}</small></div>
        <div><b>계절주기</b><span><code>{selectedModel === 'SARIMA' ? '계절주기 후보 13·26·52주' : summary.structure}</code></span><small>{selectedModel === 'SARIMA' ? 'A센터: 13·26·52주 탐색 · B센터: 13·26주 탐색' : summary.structureNote}</small></div>
        <div><b>구조 결정</b><span><code>{selectedModel === 'SARIMA' ? '센터×상품별 AICc 최소 구조 선택' : summary.decision[0]}</code></span><small>{selectedModel === 'SARIMA' ? '적합도와 복잡도를 고려해 계절구조 결정' : summary.decision[1]}</small></div>
      </div>
      {selectedModel === 'SARIMA' && <p className="sm-selection-reason">계절성 반영 · B센터 단기예측 안정화 · 외생변수 모델의 수치 불안정 회피</p>}
    </div>
  );
}

const MODE_LABEL = { search: '탐색', fixed: '고정', auto: '자동 결정' };

function ModeTag({ mode }) {
  return <span className={`mdl-mode-tag mdl-mode-${mode}`}>{MODE_LABEL[mode] || mode}</span>;
}

function SourceBlock({ title, data }) {
  return (
    <div className="mdl-source-group">
      <div className="mdl-source-title">{title}</div>
      <div className="mdl-source-meta"><b>출처</b> · {data.source}<br /><b>참고 이유</b> · {data.reason}</div>
      {data.rows.length > 0 ? (
        <table className="mdl-table">
          <thead><tr><th>파라미터</th><th>목적</th><th>{title === '선행연구' ? '선행연구 범위' : '공식 기본값 / 범위'}</th></tr></thead>
          <tbody>{data.rows.map((r) => <tr key={r.param}><td><code>{r.param}</code></td><td>{r.purpose}</td><td>{r.range}</td></tr>)}</tbody>
        </table>
      ) : <div className="mdl-source-none">명시적 범위 없음</div>}
    </div>
  );
}

function statPerfRows(holdoutPerf, model) {
  return BIAS_HORIZONS.map((h) => ({ horizon: h, entry: statPerfEntry(holdoutPerf, h.value, model) }));
}

function SarimaDetailModal({ onClose }) {
  const sections = [
    {
      id: 'overview', heading: '개요',
      body: <div className="sm-stat-detail-uniform"><table className="mdl-table"><tbody>
        <tr><th>모델</th><td><b>SARIMA</b></td></tr>
        <tr><th>설명</th><td>상품별 과거 주간 판매량의 <b>비계절 패턴과 반복되는 계절 패턴을 함께 반영</b>하는 시계열 모델</td></tr>
        <tr><th>모델 단위</th><td><b>센터 × 상품별 개별 모델</b></td></tr>
        <tr><th>외부변수</th><td>사용하지 않음</td></tr>
        <tr><th>예측 시점</th><td><b>1주 · 2주 · 4주 후</b></td></tr>
        <tr><th>역할</th><td><b>통계모델 대표모델</b></td></tr>
      </tbody></table></div>,
    },
    {
      id: 'variables', heading: '사용 변수',
      body: <div className="sm-stat-detail-uniform"><table className="mdl-table"><thead><tr><th>컬럼명</th><th>컬럼 설명</th><th>사용 방식</th></tr></thead><tbody><tr><td><code>qty_log1p</code></td><td>주간 판매수량을 <code>log(판매수량 + 1)</code>로 변환한 값</td><td>SARIMA 입력 시계열</td></tr></tbody></table></div>,
    },
    {
      id: 'preprocessing', heading: '전처리',
      body: <div className="sm-stat-detail-uniform"><table className="mdl-table"><thead><tr><th>단계</th><th>처리 내용</th></tr></thead><tbody>
        <tr><td>로그 변환</td><td>주간 판매수량을 <code>log(판매수량 + 1)</code>로 변환하여 <code>qty_log1p</code> 생성</td></tr>
        <tr><td>변환 목적</td><td>판매수량이 0인 경우도 로그 변환할 수 있도록 하고, 큰 판매량 값의 영향을 완화</td></tr>
        <tr><td>예측값 복원</td><td>모델의 로그 단위 예측값을 역변환하여 <b>원래 판매수량 단위로 복원</b></td></tr>
      </tbody></table></div>,
    },
    {
      id: 'parameters', heading: '파라미터',
      body: <div className="sm-stat-detail-uniform">
        <table className="mdl-table"><thead><tr><th>파라미터</th><th>의미</th><th>실제 적용</th></tr></thead><tbody>
          <tr><td><code>p</code></td><td>비계절 자기회귀(AR) 차수</td><td><b>ARIMA 단계에서 상품별로 확정한 값 사용</b></td></tr>
          <tr><td><code>d</code></td><td>비계절 차분 차수</td><td><b>ARIMA 단계에서 상품별로 확정한 값 사용</b></td></tr>
          <tr><td><code>q</code></td><td>비계절 이동평균(MA) 차수</td><td><b>ARIMA 단계에서 상품별로 확정한 값 사용</b></td></tr>
          <tr><td><code>P</code></td><td>계절 자기회귀 차수</td><td><code>{'{0, 1}'}</code> 탐색</td></tr>
          <tr><td><code>D</code></td><td>계절 차분 차수</td><td><b>OCSB 검정으로 결정, 최대 1</b></td></tr>
          <tr><td><code>Q</code></td><td>계절 이동평균 차수</td><td><code>{'{0, 1}'}</code> 탐색</td></tr>
          <tr><td><code>m</code></td><td>계절주기</td><td><code>{'{13, 26, 52}'}</code>주 탐색</td></tr>
          <tr><td>최종 구조</td><td><code>(p,d,q)(P,D,Q,m)</code></td><td>후보 중 <b>AICc가 가장 작은 구조 선택</b></td></tr>
          <tr><td>52주 적용 조건</td><td><code>m=52</code> 탐색 여부</td><td><b>학습 이력이 104주 이상인 경우에만 후보에 포함</b></td></tr>
          <tr><td>구조 선택 단위</td><td>파라미터 적용 범위</td><td><b>센터 × 상품별로 개별 선택</b></td></tr>
        </tbody></table>
        <div className="mdl-sub-hd">모델 적합 설정</div>
        <table className="mdl-table"><thead><tr><th>설정</th><th>값</th><th>의미</th></tr></thead><tbody>
          <tr><td><code>simple_differencing</code></td><td><code>True</code></td><td><code>d</code>, <code>D</code>에 따른 차분을 적용한 시계열을 사용해 모델 적합</td></tr>
          <tr><td><code>enforce_stationarity</code></td><td><code>False</code></td><td>AR 파라미터에 정상성 조건을 강제로 제한하지 않음</td></tr>
          <tr><td><code>enforce_invertibility</code></td><td><code>False</code></td><td>MA 파라미터에 가역성 조건을 강제로 제한하지 않음</td></tr>
        </tbody></table>
      </div>,
    },
    {
      id: 'performance', heading: '성능',
      body: <div className="sm-stat-detail-uniform"><table className="mdl-table mdl-performance-table"><thead><tr><th>예측 시점</th><th>WAPE</th><th>Bias</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>
        <tr><td><b>1주 후</b></td><td>101.18%</td><td>-32.55%</td><td>3.11</td><td>39.41</td></tr>
        <tr><td><b>2주 후</b></td><td>95.74%</td><td>-40.78%</td><td>2.95</td><td>38.20</td></tr>
        <tr><td><b>4주 후</b></td><td>101.04%</td><td>-36.57%</td><td>3.12</td><td>38.47</td></tr>
      </tbody></table></div>,
    },
  ];
  return <ModelDetailModal title="SARIMA" typeLabel="통계모델" statusBadge={{ tone: 'final', text: '최종 선정' }} sections={sections} onClose={onClose} />;
}

function DetailModal({ model, holdoutPerf, onClose }) {
  const meta = MODEL_META[model];
  const config = STAT_MODEL_DETAIL[model];
  if (!meta || !config) return null;

  const status = statStatus(model);
  const { structure, variableGroups, preprocessing, paramRationale, actualParams, finalParams, selectionCriteria } = config;
  const summary = modelUiSummary(model, meta);
  const preprocessingRows = detailPreprocessing(preprocessing);
  const parameterRows = detailParameters(model, config, summary);

  const variableCount = variableGroups.reduce((sum, group) => sum + group.items.length, 0);
  const variableTable = (group) => <table className="mdl-table"><thead><tr><th>컬럼명</th><th>컬럼 설명</th><th>활용 목적</th></tr></thead><tbody>{group.items.map((item) => <tr key={item.col}><td><code>{item.col}</code></td><td>{item.meaning}</td><td>{item.purpose}</td></tr>)}</tbody></table>;
  const priorByParam = Object.fromEntries(paramRationale.priorResearch.rows.map((row) => [row.param, row.range]));
  const officialByParam = Object.fromEntries(paramRationale.officialDocs.rows.map((row) => [row.param, row.range]));
  const sections = [
    {
      id: 'overview', heading: '개요',
      body: <div className="sm-stat-detail-uniform"><table className="mdl-table"><tbody>
        <tr><th>모델</th><td>{meta.label}</td></tr>
        <tr><th>설명</th><td>{summary.description}</td></tr>
        <tr><th>모델 단위</th><td>센터 × 상품별 개별 모델</td></tr>
        <tr><th>기본 구조</th><td>{summary.baseStructure}</td></tr>
        {summary.exog.length > 0 && <tr><th>추가 외부변수</th><td><code>{summary.exog.join(' · ')}</code></td></tr>}
        {summary.exog.length === 0 && <tr><th>외부변수</th><td>사용하지 않음</td></tr>}
        <tr><th>예측 시점</th><td>1주 · 2주 · 4주 후</td></tr>
      </tbody></table></div>,
    },
    {
      id: 'variables', heading: '사용 변수',
      body: <div className="sm-stat-detail-uniform"><table className="mdl-table"><thead><tr><th>컬럼명</th><th>컬럼 설명</th><th>모델에서의 역할</th></tr></thead><tbody>{variableGroups.flatMap((group) => group.items.map((item) => <tr key={`${group.group}-${item.col}`}><td><code>{item.col}</code></td><td>{item.col === 'qty_log1p' ? '주간 판매수량을 log(판매수량 + 1)로 변환한 값' : item.meaning}</td><td>{variableRole(item.col)}</td></tr>))}</tbody></table></div>,
    },
    {
      id: 'preprocessing', heading: '전처리·예외처리',
      body: <div className="sm-stat-detail-uniform">
        <table className="mdl-table">
          <thead><tr><th>대상</th><th>실제 처리</th><th>적용 이유</th></tr></thead>
          <tbody>{preprocessingRows.map((p) => <tr key={`${p.target}-${p.method}`}><td>{p.target}</td><td>{p.method}</td><td>{p.reason}</td></tr>)}</tbody>
        </table>
      </div>,
    },
    {
      id: 'parameters', heading: '파라미터',
      body: <div className="sm-stat-detail-uniform">
        <table className="mdl-table">
          <thead><tr><th>파라미터</th><th>의미</th><th>실제 적용</th></tr></thead>
          <tbody>{parameterRows.map((p) => <tr key={p[0]}><td><code>{p[0]}</code></td><td>{p[1]}</td><td>{p[2]}</td></tr>)}</tbody>
        </table>
      </div>,
    },
    {
      id: 'performance', heading: '성능',
      body: <div className="sm-stat-detail-uniform">
        <table className="mdl-table mdl-performance-table">
          <thead><tr><th>예측시점</th><th>WAPE(%)</th><th>Bias(%)</th><th>MAE</th><th>RMSE</th></tr></thead>
          <tbody>{statPerfRows(holdoutPerf, model).map(({ horizon, entry }) => (
            <tr key={horizon.value}>
              <td>{horizon.label}</td>
              {entry
                ? <><td>{fmtDetailMetric(entry.wape)}</td><td>{fmtDetailMetric(entry.bias)}</td><td>{fmtDetailMetric(entry.mae)}</td><td>{fmtDetailMetric(entry.rmse)}</td></>
                : <td colSpan={4} className="mdl-perf-unavailable">{holdoutPerf ? 'artifact에 없음' : '불러오는 중...'}</td>}
            </tr>
          ))}</tbody>
        </table>
        {EXCLUDED_MODELS.includes(model) && <div className="mdl-note">이 모델은 일부 SKU의 exog 계수 발산으로 지표가 극단값이 되어, 정상 비교 대상에서 제외되었습니다(값은 artifact 원본 그대로 표시).</div>}
      </div>,
    },
  ];

  return <ModelDetailModal title={meta.label} typeLabel={config.typeLabel} statusBadge={status} sections={sections} onClose={onClose} />;
}
