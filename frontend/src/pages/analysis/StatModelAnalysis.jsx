import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { api } from '../../api/client.js';
import EChart from '../../components/charts/EChart.jsx';
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
  if (v === null || v === undefined) return '-';
  const av = Math.abs(v);
  if (av >= 1000) return v.toExponential(2).replace('e+', 'e');
  return v.toFixed(1);
}

function fmtTooltip(v) {
  if (v === null || v === undefined) return '-';
  const av = Math.abs(v);
  if (av >= 1000) return v.toExponential(2).replace('e+', 'e');
  return v.toFixed(2);
}

function fmtAxisLabel(v) {
  if (Math.abs(v) >= 1000) return v.toExponential(1).replace('e+', 'e');
  return String(v);
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
const SEASONAL_PARAM = {
  range: '계절주기 m ∈ {13, 26, 52} (판매 이력이 104주 미만이면 m=52 제외), 계절 (P,Q) 각 0~1, 계절 차분 D는 OCSB 검정으로 결정 — 비계절 (p,d,q)는 ARIMA 탐색 결과를 그대로 상속',
  method: 'm 후보별로 (P,Q) 조합을 적합해 AICc 비교',
  criterion: 'AICc 최소 계절 구조 선택',
  nonConvergence: '최적 후보가 수렴하지 않으면 수렴한 후보 중 AICc 최소 구조로 대체',
};
const EXOG_ONLY_PARAM = {
  range: '(p,d,q)는 ARIMA 탐색 결과를 그대로 사용, 외생변수만 추가로 적합',
  method: '동일 (p,d,q) 구조에 exog 블록을 추가해 재적합 (별도 구조 탐색 없음)',
  criterion: 'ARIMA_S0에서 이미 선정된 구조를 상속',
  nonConvergence: 'ARIMA_S0과 동일한 수렴 판정 기준 적용',
};

const MODEL_META = {
  ARIMA_S0: {
    label: 'ARIMA', oneLiner: '외생변수·계절성 없이 판매 이력만으로 학습하는 기준 모델',
    seasonal: false, exog: false,
    columns: [QTY_COLUMN],
    structureEasy: 'ARIMA (계절성·외생변수 없음)',
    usedInfo: '판매이력만 사용 (계절패턴·외생변수 없음)',
    why: '외생변수·계절성이 전혀 없는 가장 단순한 baseline. 계절성/외생변수를 추가했을 때의 개선 효과를 비교하기 위한 기준점으로 실험',
    paramSearch: ARIMA_PARAM,
    paramShort: '과거 패턴 후보 탐색 → 수렴한 후보 중 정보기준이 가장 좋은 구조 선택',
    verdict: { selected: false, reason: '외생변수·계절성이 없는 baseline. SARIMA 대비 정확도가 낮아 대표모델로 미선정' },
  },
  SARIMA: {
    label: 'SARIMA', oneLiner: '반복되는 계절 패턴(명절 등)을 반영한 통계 대표모델',
    seasonal: true, exog: false,
    columns: [QTY_COLUMN],
    structureEasy: 'ARIMA + 계절 구조',
    usedInfo: '판매이력 + 계절패턴 (외생변수 없음)',
    why: '판매 데이터에 존재하는 주기적 계절 패턴(명절 등 반복 수요)을 반영했을 때 baseline 대비 얼마나 개선되는지 확인하기 위해 실험',
    paramSearch: SEASONAL_PARAM,
    paramShort: '계절주기 후보(13/26/52주) 탐색 → 정상 수렴 확인 → AICc 최소 구조 선택',
    verdict: { selected: true, reason: '외생변수 없이 계절성만 반영해도 h1~h4 전 구간에서 WAPE가 안정적 → 통계 대표모델로 선정' },
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
    structureEasy: 'ARIMA + 경제지표',
    usedInfo: '판매이력 + 경제지표 외생변수',
    why: '소비자심리지수·물가지수 같은 경제지표가 판매량 예측에 도움이 되는지 확인하기 위해 실험',
    paramSearch: EXOG_ONLY_PARAM,
    paramShort: 'ARIMA 구조 유지 → 경제지표를 추가해 학습',
    verdict: { selected: false, reason: '일부 SKU에서 계수가 발산해 WAPE가 극단적으로 커지는 경우가 많아 미선정' },
  },
  ARIMAX_S2: {
    label: 'ARIMAX-S2', oneLiner: 'COVID 시기 수요 변화를 추가한 ARIMA 실험 모델',
    seasonal: false, exog: true,
    columns: [
      QTY_COLUMN,
      { col: 'covid_flag', meaning: '코로나19 관련 기간 여부 플래그', purpose: '코로나 시기의 이례적 수요 변화를 반영' },
    ],
    structureEasy: 'ARIMA + COVID 지표',
    usedInfo: '판매이력 + COVID 외생변수',
    why: '코로나19 시기의 수요 변화를 외생변수로 반영했을 때 예측이 개선되는지 확인하기 위해 실험',
    paramSearch: EXOG_ONLY_PARAM,
    paramShort: 'ARIMA 구조 유지 → COVID 지표를 추가해 학습',
    verdict: { selected: false, reason: 'ARIMA baseline 대비 큰 개선이 없어 미선정' },
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
    structureEasy: 'ARIMA + 공휴일 지표',
    usedInfo: '판매이력 + 공휴일 외생변수',
    why: '설·추석 등 공휴일 전후의 판매 변화를 외생변수로 반영했을 때 예측이 개선되는지 확인하기 위해 실험',
    paramSearch: EXOG_ONLY_PARAM,
    paramShort: 'ARIMA 구조 유지 → 공휴일 지표를 추가해 학습',
    verdict: { selected: false, reason: 'ARIMA baseline 대비 큰 개선이 없어 미선정' },
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
    structureEasy: 'ARIMA + 경제·COVID·공휴일 지표 전체',
    usedInfo: '판매이력 + 경제·COVID·공휴일 외생변수 전체(7종)',
    why: '경제·COVID·공휴일 외생변수를 모두 결합했을 때의 효과를 확인하기 위해 실험',
    paramSearch: EXOG_ONLY_PARAM,
    paramShort: 'ARIMA 구조 유지 → 외부 지표를 함께 학습',
    verdict: { selected: false, reason: '외생변수가 많아 계수 발산 빈도가 가장 높아 미선정' },
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
    structureEasy: 'ARIMA + 계절 구조 + 경제·COVID·공휴일 지표 전체',
    usedInfo: '판매이력 + 계절패턴 + 경제·COVID·공휴일 외생변수 전체(7종)',
    why: '계절성과 모든 외생변수를 함께 결합했을 때 SARIMA보다 더 나은지 확인하기 위해 실험',
    paramSearch: { range: 'SARIMA에서 결정한 (p,d,q), (P,D,Q,m)과 추세를 상속', method: '확정 구조에 경제3·COVID1·공휴일3 외생변수를 함께 넣어 계수 재추정', criterion: 'SARIMA의 AICc 최소 계절 구조 상속 (별도 구조 탐색 없음)', nonConvergence: '적합 결과의 수렴 상태를 별도로 확인' },
    paramShort: 'SARIMA 구조 유지 → 외부 지표를 함께 학습',
    verdict: { selected: false, reason: '계절성 + 외생변수를 결합했지만 SARIMA 대비 WAPE가 더 높고 변동폭이 커서 미선정' },
  },
};

export default function StatModelAnalysis() {
  const [center, setCenter] = useState('ALL');
  const [metric, setMetric] = useState('WAPE');
  const [selectedModel, setSelectedModel] = useState('SARIMA');
  const [detailOpen, setDetailOpen] = useState(false);

  const [heatmap, setHeatmap] = useState(null);
  const [profile, setProfile] = useState(null);
  const [centerCompare, setCenterCompare] = useState(null);

  const [heatmapError, setHeatmapError] = useState(null);
  const [profileError, setProfileError] = useState(null);
  const [compareError, setCompareError] = useState(null);

  useEffect(() => {
    let ignore = false;
    setHeatmapError(null);
    setHeatmap(null);
    api.getStatHeatmap({ center, metric })
      .then((d) => { if (!ignore) setHeatmap(d); })
      .catch((e) => { if (!ignore) { setHeatmapError(e.message); setHeatmap(null); } });
    return () => { ignore = true; };
  }, [center, metric]);

  useEffect(() => {
    let ignore = false;
    setProfileError(null);
    setProfile(null);
    api.getStatHorizonProfile({ center, metric, include_extreme: false })
      .then((d) => { if (!ignore) setProfile(d); })
      .catch((e) => { if (!ignore) { setProfileError(e.message); setProfile(null); } });
    return () => { ignore = true; };
  }, [center, metric]);

  useEffect(() => {
    let ignore = false;
    setCompareError(null);
    setCenterCompare(null);
    Promise.all(CENTERS.map(async ({ value }) => {
      const response = await api.getStatHeatmap({ center: value, metric });
      const row = response.rows.find((r) => r.model === selectedModel);
      return { center: value, isExtreme: row?.is_extreme === true, horizons: response.horizons, values: response.horizons.map((h) => row?.values[h] ?? null) };
    }))
      .then((series) => { if (!ignore) setCenterCompare({ model: selectedModel, label: MODEL_META[selectedModel].label, metric, horizons: series[0].horizons, series }); })
      .catch((e) => { if (!ignore) { setCompareError(e.message); setCenterCompare(null); } });
    return () => { ignore = true; };
  }, [selectedModel, metric]);

  return (
    <>
      <div className="az-page-hd">
        <div>
          <h2>02 통계모델 분석</h2>
          <p>통계모델 중 어떤 구조가 가장 적합했는가</p>
        </div>
        <div className="az-filter-bar">
          <div className="az-filter-group">
            <span className="az-filter-label">Center</span>
            <select className="az-filter-select" value={center} onChange={(e) => setCenter(e.target.value)}>
              {CENTERS.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
            </select>
          </div>
          <div className="az-filter-group">
            <span className="az-filter-label">Metric</span>
            <select className="az-filter-select" value={metric} onChange={(e) => setMetric(e.target.value)}>
              {METRICS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="az-grid az-grid-1-2-1 sm-row-2-1">
        <FamilyTreeCard selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <HeatmapCard data={heatmap} error={heatmapError} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <DetailCard selectedModel={selectedModel} onOpenDetail={() => setDetailOpen(true)} />
      </div>

      <div className="az-grid az-grid-3 sm-row-3">
        <ProfileCard data={profile} error={profileError} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <BiasMapCard center={center} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <CenterCompareCard data={centerCompare} error={compareError} metric={metric} selectedModel={selectedModel} />
      </div>

      {detailOpen && <DetailModal model={selectedModel} onClose={() => setDetailOpen(false)} />}
    </>
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

const MODEL_COLOR = Object.fromEntries(
  FAMILY_NODES.filter((n) => n.model).map((n, i) => [n.model, PALETTE[i % PALETTE.length]]),
);

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
      symbolSize: n.hub ? [50, 24] : [46, 22],
      itemStyle: n.hub
        ? { color: '#f8fafc', borderColor: '#cbd5e1', borderType: 'dashed', borderWidth: 1 }
        : {
            color: isSelected ? '#2563eb' : '#eff6ff',
            borderColor: isSelected ? '#1d4ed8' : '#bfdbfe',
            borderWidth: isSelected ? 2 : 1,
          },
      label: {
        show: true,
        fontSize: 9,
        lineHeight: 10.5,
        color: n.hub ? '#94a3b8' : (isSelected ? '#fff' : '#1e40af'),
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
    left: `calc(${TREE_GRID.left}px + (100% - ${TREE_GRID.left + TREE_GRID.right}px) * ${n.x / 100} - 23px)`,
    top: `${TREE_GRID.top + (n.y / 100) * (TREE_GRID.height - TREE_GRID.top - TREE_GRID.bottom) - 11}px`,
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
        <h3>통계모델 계보도</h3>
        <p>ARIMA 계열 모델 구조와 비교 흐름</p>
      </div>
      <div className="sm-tree-wrap">
        <EChart
          height={300}
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
    </div>
  );
}

function HeatmapCard({ data, error, selectedModel, onSelectModel }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>모델별 예측시점 성능</h3>
        <p>모델별 · 예측기간별 {data?.metric || ''} 비교</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <>
          <EChart
            height={280}
            option={buildHeatmapOption(data, selectedModel)}
            onEvents={{
              click: (p) => { if (p.data?.model) onSelectModel(p.data.model); },
            }}
          />
          <div className="sm-scale-bar">
            <span>{data.metric === 'Bias' ? '음수(과소예측)' : '낮음(우수)'}</span>
            <span className="sm-scale-gradient" />
            <span>{data.metric === 'Bias' ? '양수(과대예측) · 0에 가까울수록 좋음' : '높음(미흡)'}</span>
          </div>
        </>
      )}
    </div>
  );
}

function buildHeatmapOption(data, selectedModel) {
  const horizons = data.horizons;
  const models = data.rows.map((r) => r.label);
  const bound = Math.max(Math.abs(data.color_scale.min), Math.abs(data.color_scale.max));
  const { min, max } = data.metric === 'Bias' ? { min: -bound, max: bound } : data.color_scale;
  const selectedIdx = data.rows.findIndex((r) => r.model === selectedModel);

  const cells = [];
  data.rows.forEach((row, yi) => {
    horizons.forEach((h, xi) => {
      const raw = row.values[h];
      if (!Number.isFinite(raw)) return;
      const clipped = Math.min(Math.max(raw, min), max);
      cells.push({ value: [xi, yi, clipped], raw, model: row.model, label: row.label });
    });
  });

  return {
    grid: { left: 96, right: 16, top: 10, bottom: 26 },
    tooltip: {
      ...LIGHT_TOOLTIP,
      formatter: (p) => `${p.data.label} · ${horizons[p.data.value[0]]}<br/>${data.metric}: ${fmtTooltip(p.data.raw)}`,
    },
    xAxis: {
      type: 'category', data: horizons,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11 },
    },
    yAxis: {
      type: 'category', data: models, inverse: true,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11 },
    },
    visualMap: { min, max, show: false, seriesIndex: 0, inRange: { color: ['#bfdbfe', '#f8fafc', '#fecaca'] } },
    series: [{
      type: 'heatmap',
      data: cells,
      label: { show: true, formatter: (p) => fmtMetric(p.data.raw), fontSize: 11, color: '#1e293b' },
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

function ProfileCard({ data, error, selectedModel, onSelectModel }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>예측시점별 성능 변화</h3>
        <p>{data ? `${metricLabel(data.metric)} · ${metricGuide(data.metric)} (극단 모델 제외)` : '지표 불러오는 중'}</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <EChart
          height={260}
          option={buildProfileOption(data, selectedModel)}
          onEvents={{
            click: (p) => {
              const s = data.series.find((x) => x.label === p.seriesName);
              if (s) onSelectModel(s.model);
            },
          }}
        />
      )}
    </div>
  );
}

function buildProfileOption(data, selectedModel) {
  return {
    grid: { left: 46, right: 16, top: 16, bottom: 40 },
    tooltip: { ...LIGHT_TOOLTIP, trigger: 'axis', formatter: (params) => `${metricLabel(data.metric)}<br/>${axisTooltipFormatter(params)}` },
    legend: { bottom: 0, textStyle: { fontSize: 10.5, color: '#64748b' } },
    color: PALETTE,
    xAxis: { type: 'category', data: data.horizons, axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b' } },
    yAxis: { type: 'value', name: metricLabel(data.metric), splitLine: { lineStyle: { color: '#f1f5f9' } }, axisLabel: { color: '#64748b', formatter: fmtAxisLabel } },
    series: data.series.map((s) => {
      const isSelected = s.model === selectedModel;
      return {
        name: s.label,
        type: 'line',
        data: s.values,
        symbol: 'circle',
        symbolSize: isSelected ? 7 : 5,
        lineStyle: { width: isSelected ? 3.5 : 1.5, opacity: isSelected ? 1 : 0.35 },
        itemStyle: { opacity: isSelected ? 1 : 0.35 },
        z: isSelected ? 10 : 1,
      };
    }),
  };
}

function BiasMapCard({ center, selectedModel, onSelectModel }) {
  const [horizon, setHorizon] = useState('h1');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;
    setError(null);
    api.getStatWapeBias({ center, horizon, include_extreme: false })
      .then((d) => { if (!ignore) setData(d); })
      .catch((e) => { if (!ignore) { setError(e.message); setData(null); } });
    return () => { ignore = true; };
  }, [center, horizon]);

  return (
    <div className="az-card">
      <div className="az-card-hd sm-card-hd-row">
        <div>
          <h3>정확도 × 편향 비교</h3>
          <p>WAPE × Bias 동시 비교</p>
        </div>
        <div className="sm-seg" role="group" aria-label="예측시점 선택">
          {BIAS_HORIZONS.map((h) => (
            <button
              key={h.value}
              type="button"
              className={`sm-seg-btn${horizon === h.value ? ' active' : ''}`}
              onClick={() => setHorizon(h.value)}
            >
              {h.label}
            </button>
          ))}
        </div>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <EChart
          height={230}
          option={buildBiasOption(data, selectedModel)}
          onEvents={{
            click: (p) => { if (p.data?.model) onSelectModel(p.data.model); },
          }}
        />
      )}
    </div>
  );
}

function buildBiasOption(data, selectedModel) {
  return {
    grid: { left: 50, right: 20, top: 16, bottom: 30 },
    tooltip: {
      ...LIGHT_TOOLTIP,
      formatter: (p) => `${p.data.modelLabel}<br/>WAPE ${fmtTooltip(p.data.value[0])} · Bias ${fmtTooltip(p.data.value[1])}`,
    },
    xAxis: { type: 'value', name: 'WAPE', axisLabel: { color: '#64748b', formatter: fmtAxisLabel }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    yAxis: { type: 'value', name: 'Bias', axisLabel: { color: '#64748b', formatter: fmtAxisLabel }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{
      type: 'scatter',
      data: data.points.map((p) => {
        const isSelected = p.model === selectedModel;
        return {
          value: [p.wape, p.bias], model: p.model, modelLabel: p.label,
          symbolSize: isSelected ? 20 : 11,
          itemStyle: {
            color: MODEL_COLOR[p.model] || '#93c5fd',
            opacity: isSelected ? 1 : 0.8,
            borderColor: isSelected ? '#fff' : 'transparent',
            borderWidth: isSelected ? 2 : 0,
          },
          label: {
            show: true,
            formatter: (pp) => pp.data.modelLabel,
            fontSize: isSelected ? 12 : 10.5,
            fontWeight: isSelected ? 700 : 500,
            color: isSelected ? '#1e293b' : '#94a3b8',
          },
        };
      }),
      labelLayout: { moveOverlap: 'shiftY' },
      markLine: {
        silent: true, symbol: 'none',
        lineStyle: { color: '#cbd5e1', type: 'dashed' },
        label: { show: false },
        data: [{ yAxis: 0 }],
      },
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
        <tbody>{['ALL', 'A', 'B'].map((center) => {
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
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>센터별 성능 비교{` (${MODEL_META[selectedModel].label})`}</h3>
        <p>{metricLabel(metric)} · {metricGuide(metric)} (1·2·4주 후)</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (data.series.some((s) => s.values.some(Number.isFinite))
        ? (needsCenterMatrix(data) ? <CenterResultMatrix data={data} /> : <EChart height={260} option={buildCenterCompareOption(data)} />)
        : <div className="az-hint">해당 지표 데이터 없음</div>)}
    </div>
  );
}

function buildCenterCompareOption(data) {
  const categories = data.series.map((s) => (s.center === 'ALL' ? 'ALL' : `${s.center}센터`));
  return {
    grid: { left: 46, right: 16, top: 30, bottom: 40 },
    tooltip: { ...LIGHT_TOOLTIP, trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: (params) => `${metricLabel(data.metric)}<br/>${axisTooltipFormatter(params)}` },
    legend: { bottom: 0, textStyle: { fontSize: 10.5, color: '#64748b' } },
    color: PALETTE,
    xAxis: { type: 'category', data: categories, axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b' } },
    yAxis: { type: 'value', name: metricLabel(data.metric), splitLine: { lineStyle: { color: '#f1f5f9' } }, axisLabel: { color: '#64748b', formatter: fmtAxisLabel } },
    series: data.horizons.map((h, hi) => ({
      name: h,
      type: 'bar',
      barMaxWidth: 22,
      label: { show: true, position: 'top', fontSize: 9.5, fontWeight: 600, color: '#64748b', formatter: (p) => fmtMetric(p.value) },
      data: data.series.map((s) => s.values[hi]),
    })),
  };
}

function DetailCard({ selectedModel, onOpenDetail }) {
  const meta = MODEL_META[selectedModel];
  if (!meta) return null;
  return (
    <button type="button" className="az-card sm-detail-card" onClick={onOpenDetail}>
      <div className="az-card-hd">
        <h3>선택 모델 요약</h3>
      </div>
      <div className="az-detail-title">{meta.label}</div>
      <div className="sm-detail-oneliner">{meta.oneLiner}</div>
      <div className="az-detail-row"><span className="az-detail-label">예측 대상</span><span className="az-detail-value">센터 × SKU별 주간 판매수량</span></div>
      <div className="az-detail-row"><span className="az-detail-label">사용 정보</span><span className="az-detail-value">{meta.usedInfo}</span></div>
      <div className="az-detail-row"><span className="az-detail-label">구조</span><span className="az-detail-value">{meta.structureEasy}</span></div>
      <div className="az-detail-row"><span className="az-detail-label">파라미터 결정</span><span className="az-detail-value">{meta.paramShort}</span></div>
      <div className="az-detail-row"><span className="az-detail-label">최종 판단</span><span className="az-detail-value">{meta.verdict.reason}</span></div>
      <span className="sm-detail-more">상세 보기 →</span>
    </button>
  );
}

function DetailModal({ model, onClose }) {
  const meta = MODEL_META[model];

  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose(); }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!meta) return null;

  return (
    <div className="sm-modal-backdrop" onClick={onClose}>
      <div className="sm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="sm-modal-hd">
          <h3>{meta.label}</h3>
          <button className="sm-modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="sm-modal-body">
          <section className="sm-modal-section">
            <h4>① 모델 개요</h4>
            <p>{meta.oneLiner}. {meta.why}</p>
          </section>

          <section className="sm-modal-section">
            <h4>② 입력 데이터 / 컬럼</h4>
            <p>{INPUT_DATA}</p>
            <p>학습: development_2021_2023.parquet · 평가: holdout_2024.parquet</p>
            <table className="sm-modal-table">
              <thead><tr><th>컬럼명</th><th>컬럼 설명</th><th>모델에서의 활용 목적</th></tr></thead>
              <tbody>
                {inputColumns(model, meta).map((c) => (
                  <tr key={c.col}><td><code>{c.col}</code></td><td>{c.meaning}</td><td>{c.purpose}</td></tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="sm-modal-section">
            <h4>③ 모델 구조 / 외생변수 구성</h4>
            <p>외생변수: {meta.exog ? meta.columns.filter((c) => c !== QTY_COLUMN).map((c) => c.col).join(" · ") : "없음"}</p>
            <p>{meta.structureEasy} — 계절성 {meta.seasonal ? '사용' : '미사용'}, 외생변수 {meta.exog ? '사용' : '미사용'}</p>
          </section>

          <section className="sm-modal-section">
            <h4>④ 파라미터 결정 방법</h4>
            <div className="sm-modal-kv"><span>탐색 범위</span><span>{meta.paramSearch.range}</span></div>
            <div className="sm-modal-kv"><span>탐색 방식</span><span>{meta.paramSearch.method}</span></div>
            <div className="sm-modal-kv"><span>선정 기준</span><span>{meta.paramSearch.criterion}</span></div>
            <div className="sm-modal-kv"><span>미수렴 처리</span><span>{meta.paramSearch.nonConvergence}</span></div>
            {model === 'SARIMA' && (
              <p className="sm-modal-note">p = 과거 판매값 영향 범위 · d = 차분 횟수 · q = 과거 예측오차 영향 범위 · m = 계절 반복주기</p>
            )}
          </section>

          <section className="sm-modal-section">
            <h4>⑤ 선택 근거</h4>
            <p className={meta.verdict.selected ? 'sm-modal-verdict sm-modal-verdict-yes' : 'sm-modal-verdict'}>
              {meta.verdict.selected ? '통계 대표모델로 선정' : '대표모델로 선정되지 않음'} — {meta.verdict.reason}
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
