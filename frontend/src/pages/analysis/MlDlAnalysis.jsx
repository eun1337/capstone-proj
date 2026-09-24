import { useEffect, useState } from 'react';
import { api } from '../../api/client.js';
import EChart from '../../components/charts/EChart.jsx';
import ModelDetailModal from '../../components/ModelDetailModal.jsx';
import { ML_DL_MODEL_DETAIL } from './mlDlModelDetailConfig.js';
import './analysis.css';
import './MlDlAnalysis.css';

const HORIZONS = [
  { value: 'h1', label: '1주 후' },
  { value: 'h2', label: '2주 후' },
  { value: 'h4', label: '4주 후' },
];
const HORIZON_KEYS = ['h1', 'h2', 'h4'];

const MLDL_MODEL_LABELS = {
  RF: 'RF', LightGBM: 'LGBM', LSTM: 'LSTM', TFT: 'TFT', Informer: 'Informer',
  'Hurdle-RF': 'H-RF', 'Hurdle-LightGBM': 'H-LGBM',
};
const LABEL_TO_MODEL = Object.fromEntries(Object.entries(MLDL_MODEL_LABELS).map(([k, v]) => [v, k]));
const horizonLabel = (h) => HORIZONS.find((item) => item.value === h)?.label || '—';

const FINAL_MODEL = 'Hurdle-LightGBM';
const FINAL_COLOR = '#16a34a';
const BEFORE_COLOR = '#dc2626';

const MODEL_COLORS = {
  RF: '#d97706', LightGBM: '#d97706',
  LSTM: '#7c3aed', TFT: '#7c3aed', Informer: '#7c3aed',
  'Hurdle-RF': '#2563eb', 'Hurdle-LightGBM': '#2563eb',
};
const LEGACY_MODELS = ['RF', 'LightGBM', 'LSTM', 'TFT', 'Informer'];

const TARGET_COL_BY_HORIZON = { h1: 'target_h1', h2: 'target_h2', h4: 'target_h4' };
const DL_MODELS = new Set(['LSTM', 'TFT', 'Informer']);

const TOOLTIP_BASE = { appendTo: 'body', confine: true, backgroundColor: '#fff', borderColor: '#cbd5e1', textStyle: { color: '#334155' }, extraCssText: 'z-index:99999;' };

function fmtMetric(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  return v.toFixed(2);
}

function niceBiasAxis(dataMin, biasBand) {
  const min = Math.floor(Math.min(dataMin, -biasBand) / 10) * 10;
  const max = biasBand + 5;
  return { min, max };
}

function niceLinearBounds(values, step) {
  const nums = values.filter((v) => v !== null && v !== undefined && Number.isFinite(v));
  if (!nums.length) return { min: 0, max: step };
  const min = Math.floor(Math.min(...nums) / step) * step;
  const max = Math.ceil(Math.max(...nums) / step) * step;
  return min === max ? { min: min - step, max: max + step } : { min, max };
}

function splitChips(text) {
  if (!text) return [];
  if (text.includes(';')) return text.split(';').map((s) => s.trim()).filter(Boolean);
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '(') depth++;
    else if (c === ')') depth--;
    else if (c === ',' && depth === 0) {
      parts.push(text.slice(start, i).trim());
      start = i + 1;
    }
  }
  parts.push(text.slice(start).trim());
  return parts.filter(Boolean);
}

function firstKeyValueSeparator(seg) {
  const eqIdx = seg.search(/(?<![<>=!])=(?!=)/);
  const colonIdx = seg.indexOf(':');
  const candidates = [eqIdx, colonIdx].filter((i) => i !== -1);
  return candidates.length ? Math.min(...candidates) : -1;
}

function parseKV(text) {
  return splitChips(text).map((seg) => {
    const idx = firstKeyValueSeparator(seg);
    if (idx === -1) return { key: null, value: seg };
    const key = seg.slice(0, idx).trim();
    if (key.length > 24 || key.split(' ').length > 3) return { key: null, value: seg };
    return { key, value: seg.slice(idx + 1).trim() };
  });
}

function KVGrid({ text }) {
  const items = parseKV(text);
  if (!items.length) return null;
  if (!items.some((it) => it.key)) return <div className="md-detail-caption">{text}</div>;
  return (
    <div className="md-kv-grid">
      {items.map((it, i) => (
        <div className={`md-kv-item${it.key ? '' : ' md-kv-span'}`} key={i}>
          {it.key && <span className="md-kv-key">{it.key}</span>}
          <span className="md-kv-value">{it.value}</span>
        </div>
      ))}
    </div>
  );
}

function InfoTip({ text }) {
  return <span className="md-info-tip" title={text}>ⓘ</span>;
}

function statusLabel(status, trial) {
  return status === 'selected' ? `selected trial (#${trial})` : `Bias 기준 미통과 · 최저 WAPE 후보 (#${trial})`;
}

function modelTypeLabel(model) {
  if (model.startsWith('Hurdle-')) return 'Hurdle(분류 + 조건부 회귀)';
  return DL_MODELS.has(model) ? '딥러닝(시계열)' : '머신러닝(회귀)';
}

export default function MlDlAnalysis() {
  const [selectedModel, setSelectedModel] = useState(FINAL_MODEL);
  const [summary, setSummary] = useState(null);
  const [summaryError, setSummaryError] = useState(null);

  useEffect(() => {
    api.getMlDlModelSummary()
      .then(setSummary)
      .catch((e) => setSummaryError(e.message));
  }, []);

  return (
    <div className="md-page">
      <div className="az-page-hd">
        <div>
          <h2>03 ML/DL 분석</h2>
          <p>2023 모델선정/CV 결과 · 기존 모델 문제 → Hurdle 도입 → 개선 → H-LGBM 선정</p>
        </div>
      </div>

      <div className="md-dashboard-grid">
        <ModelScatterCard summary={summary} error={summaryError} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
        <LegacyBiasTrendCard summary={summary} error={summaryError} />
        <HurdleIntroCard />
        <BeforeAfterCard summary={summary} error={summaryError} />
        <HurdleCompareCard summary={summary} error={summaryError} />
        <FinalModelCard summary={summary} error={summaryError} selectedModel={selectedModel} onSelectModel={setSelectedModel} />
      </div>
    </div>
  );
}

function ModelScatterCard({ summary, error, selectedModel, onSelectModel }) {
  const [horizon, setHorizon] = useState('h1');
  const [comparisonView, setComparisonView] = useState('legacy');
  const entries = summary ? summary.entries.filter((e) => e.horizon === horizon && (comparisonView === 'legacy' ? LEGACY_MODELS.includes(e.model) : true)) : null;

  return (
    <div className="az-card">
      <div className="az-card-hd md-scatter-hd">
        <div>
          <h3>1. 초기 5개 모델 + 후속 Hurdle 검증 결과 <span className="md-dev-badge">A센터 개발검증</span>
            <InfoTip text="selected = Bias guardrail(±20%) 통과 trial · 참고 후보 = 미통과 시 최저 WAPE trial. 모델마다 Trial 수가 달라 우열 비교 근거로 쓰지 않습니다." />
          </h3>
          <p>A센터 2023 · 4개 검증구간 통합 결과</p>
        </div>
        <div className="md-card-controls">
          <select className="md-compact-select" aria-label="모델 비교 범위 선택" value={comparisonView} onChange={(event) => setComparisonView(event.target.value)}><option value="legacy">기존 모델</option><option value="hurdle">Hurdle 비교</option></select>
          <select className="md-compact-select" aria-label="예측시점 선택" value={horizon} onChange={(event) => setHorizon(event.target.value)}>{HORIZONS.map((h) => <option key={h.value} value={h.value}>{h.label}</option>)}</select>
        </div>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !entries && <div className="az-hint">불러오는 중...</div>}
      {!error && entries && (
        <>
          <EChart
            fill
            option={buildModelScatterOption(entries, selectedModel, summary.entries, comparisonView)}
            onEvents={{ click: (p) => { if (p.data?.model) onSelectModel(p.data.model); } }}
          />
          <div className="md-legend-row">
            <span className="md-legend-chip"><i style={{ background: MODEL_COLORS.RF }} />ML</span>
            <span className="md-legend-chip"><i style={{ background: MODEL_COLORS.LSTM }} />DL</span>
            {comparisonView === 'hurdle' && <span className="md-legend-chip"><i style={{ background: MODEL_COLORS['Hurdle-RF'] }} />Hurdle</span>}

          </div>
          <div className="md-caption" title="초록 안정 구간은 Bias ±20% 이내이면서 전체 주차 H-LGBM 최대 WAPE 이하인 영역입니다.">낮은 오차 · 작은 편향 방향이 안정적 · 허용 Bias ±20%</div>
        </>
      )}
    </div>
  );
}

function buildModelScatterOption(entries, selectedModel, allEntries, comparisonView) {
  const biasValues = allEntries.map((e) => e.bias);
  const wapeValues = allEntries.map((e) => e.wape);
  const { min: yMin, max: yMax } = niceBiasAxis(Math.min(...biasValues, -20), 20);
  const { min: xMin, max: xMax } = niceLinearBounds(wapeValues, 5);

  const finalCandidates = allEntries.filter((entry) => entry.model === FINAL_MODEL && entry.bias_pass && Number.isFinite(entry.wape));
  const finalRegion = comparisonView === 'hurdle' && finalCandidates.length ? { left: -20, right: 20, bottom: xMin, top: Math.max(...finalCandidates.map((entry) => entry.wape)) } : null;
  const mainData = entries.map((e) => {
    const isFinal = e.model === FINAL_MODEL;
    const isLegacy = LEGACY_MODELS.includes(e.model);
    const baseColor = comparisonView === 'hurdle' && isLegacy ? '#cbd5e1' : MODEL_COLORS[e.model];
    const isRef = e.status === 'reference_only';
    const showLabel = true;
    return {
      value: [e.bias, e.wape],
      model: e.model, modelLabel: e.label, trial: e.trial, status: e.status, bias_pass: e.bias_pass,
      symbolSize: 13,
      itemStyle: { color: baseColor, opacity: comparisonView === 'hurdle' && isLegacy ? .55 : 1, borderColor: baseColor, borderWidth: 1 },
      label: { show: showLabel, formatter: e.label, position: 'top', distance: 8, fontSize: 11.5, fontWeight: 700, color: baseColor, backgroundColor: 'rgba(255,255,255,.85)', padding: [2, 3] },
    };
  });

  const series = [{
    type: 'scatter',
    data: mainData,
    labelLayout: (params) => ({ x: params.rect.x + (params.dataIndex % 2 ? 24 : -12), y: params.rect.y - 16 - Math.floor(params.dataIndex / 2) * 10, align: params.dataIndex % 2 ? 'left' : 'right', moveOverlap: 'shiftY', hideOverlap: false }),
    labelLine: { show: true, lineStyle: { color: '#94a3b8' } },
    markArea: {
      silent: true, itemStyle: { color: 'rgba(22,163,74,0.12)' },
      label: { show: false },
      data: [[{ xAxis: -20, itemStyle: { color: 'rgba(22,163,74,.035)' } }, { xAxis: 20 }], ...(finalRegion ? [[{ xAxis: finalRegion.left, yAxis: finalRegion.bottom, label: { formatter: '낮은 오차 · 작은 편향\n안정 구간', position: 'insideBottom', color: '#166534', fontWeight: 700 }, itemStyle: { color: { type: 'linear', x: 0, y: 1, x2: 0, y2: 0, colorStops: [{ offset: 0, color: 'rgba(22,163,74,.28)' }, { offset: 1, color: 'rgba(187,247,208,.10)' }] } } }, { xAxis: finalRegion.right, yAxis: finalRegion.top }]] : [])],
    },
    markLine: { silent: true, symbol: 'none', label: { show: false }, lineStyle: { color: '#94a3b8', type: 'dashed' }, data: [{ xAxis: 0 }] },
  }];

  return {
    grid: { left: 46, right: 16, top: 20, bottom: 26 },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => {
        const d = p.data;
        if (!d || d.model === undefined) return '';
        return `<strong>${d.modelLabel}</strong><br/>WAPE ${fmtMetric(d.value[1])}% · Bias ${fmtMetric(d.value[0])}%`
          + `<br/>${statusLabel(d.status, d.trial)}`
          + `<br/>${d.bias_pass ? 'Bias 기준 통과' : 'Bias 기준 미통과'}`;
      },
    },
    xAxis: { type: 'value', name: 'Bias (%)', nameGap: 4, nameTextStyle: { fontSize: 10, color: '#94a3b8' }, min: yMin, max: yMax, interval: 10, axisLabel: { color: '#64748b', fontSize: 10.5 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    yAxis: { type: 'value', name: 'WAPE (%)', min: xMin, max: xMax, interval: 5, axisLabel: { color: '#64748b', fontSize: 10.5 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series,
  };
}

const LEGACY_HORIZON_COLORS = { h1: '#cbd5e1', h2: '#94a3b8', h4: '#475569' };
const LEGACY_HORIZON_SYMBOLS = { h1: 'circle', h2: 'diamond', h4: 'triangle' };

function LegacyBiasTrendCard({ summary, error }) {
  const [comparisonView, setComparisonView] = useState('legacy');
  const entries = summary ? summary.entries : null;
  return (
    <div className="az-card">
      <div className="az-card-hd md-card-title-actions">
        <div><h3>2. 예측시점별 Bias 비교(기존 모델 + Hurdle) <span className="md-dev-badge">A센터 개발검증</span></h3><p>A센터 개발검증 · 4개 검증구간 통합 Bias</p></div>
        <div className="md-card-controls"><select className="md-compact-select" aria-label="모델 비교 범위 선택" value={comparisonView} onChange={(event) => setComparisonView(event.target.value)}><option value="legacy">기존 모델</option><option value="hurdle">Hurdle 비교</option></select></div>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !entries && <div className="az-hint">불러오는 중...</div>}
      {!error && entries && (
        <>
          <EChart fill option={buildLegacyBiasOption(entries, comparisonView)} />
          <div className="md-legend-row">
            {HORIZON_KEYS.map((h) => (
              <span className="md-legend-chip" key={h}><i style={{ background: LEGACY_HORIZON_COLORS[h] }} />{horizonLabel(h)}</span>
            ))}
          </div>
          <div className="md-caption">왼쪽 과소예측 · 오른쪽 과대예측 · 0%에 가까울수록 좋음</div>
        </>
      )}
    </div>
  );
}

function buildLegacyBiasOption(entries, comparisonView) {
  const byModel = {};
  entries.forEach((e) => { (byModel[e.model] ??= {})[e.horizon] = e; });
  const visibleModels = comparisonView === 'legacy' ? LEGACY_MODELS : [...LEGACY_MODELS, 'Hurdle-RF', FINAL_MODEL];
  const categories = visibleModels.filter((m) => byModel[m]).map((m) => MLDL_MODEL_LABELS[m]);

  const biasValues = entries.map((e) => e.bias);
  const xMin = Math.floor((Math.min(...biasValues.filter(Number.isFinite), -20) - 2) / 5) * 5;
  const xMax = Math.ceil((Math.max(...biasValues.filter(Number.isFinite), 0) + 2) / 5) * 5;

  const perModelLines = categories.map((label) => {
    const m = LABEL_TO_MODEL[label];
    return {
      type: 'line', silent: true, symbol: 'none', z: 1,
      lineStyle: { color: '#cbd5e1', width: 1.4 },
      data: HORIZON_KEYS.filter((h) => byModel[m][h]).map((h) => [byModel[m][h].bias, label]),
    };
  });

  const horizonSeries = HORIZON_KEYS.map((h) => ({
    name: horizonLabel(h),
    type: 'scatter', symbol: 'circle', symbolSize: 10,
    itemStyle: { color: LEGACY_HORIZON_COLORS[h] },
    data: categories.map((label) => {
      const m = LABEL_TO_MODEL[label];
      const e = byModel[m][h];
      const isLegacy = LEGACY_MODELS.includes(m);
      const color = comparisonView === 'hurdle' && isLegacy ? '#cbd5e1' : MODEL_COLORS[m];
      return e ? { value: [e.bias, label], model: m, horizon: h, wape: e.wape, trial: e.trial, status: e.status, itemStyle: { color, opacity: comparisonView === 'hurdle' && isLegacy ? .55 : { h1: .6, h2: .8, h4: 1 }[h], borderColor: '#fff', borderWidth: 1 } } : null;
    }).filter(Boolean),
  }));

  horizonSeries[0].markArea = { silent: true, itemStyle: { color: 'rgba(22,163,74,0.12)' }, data: [[{ xAxis: -20 }, { xAxis: 20 }]] };


  return {
    grid: { left: 52, right: 6, top: 12, bottom: 22 },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => {
        const d = p.data;
        if (!d || d.model === undefined) return '';
        return `<strong>${MLDL_MODEL_LABELS[d.model]}</strong> · ${horizonLabel(d.horizon)}<br/>WAPE ${fmtMetric(d.wape)}% · Bias ${fmtMetric(d.value[0])}%<br/>${statusLabel(d.status, d.trial)}`;
      },
    },
    xAxis: { type: 'value', name: 'Bias (%)', nameGap: 2, nameTextStyle: { fontSize: 10, color: '#94a3b8' }, min: xMin, max: xMax, interval: 10, axisLabel: { color: '#64748b', fontSize: 10.5 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    yAxis: { type: 'category', inverse: true, data: categories, axisLabel: { color: '#334155', fontSize: 11, fontWeight: 600 }, axisLine: { lineStyle: { color: '#e2e8f0' } } },
    series: horizonSeries,
  };
}

function HurdleIntroCard() {
  const [open, setOpen] = useState(false);
  return (
    <div className="az-card md-hurdle-intro-card" role="button" tabIndex={0} aria-haspopup="dialog" onClick={() => setOpen(true)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(true); } }}>
      <div className="az-card-hd">
        <h3>3. 과소예측 개선을 위한 Hurdle 구조</h3>
        <p>간헐적 수요를 반영하기 위해 판매 발생 여부와 판매량을 분리</p>
      </div>
      <div className="md-hurdle-rationale"><strong>주간 0수요 비율 · A 73.85% · B 65.03%</strong><span>0수요가 많은 구조 → 판매 발생 여부와 발생 시 판매량 분리</span></div>
      <div className="md-hurdle-diagram md-hurdle-diagram-flow">
        <div className="md-hurdle-box md-hurdle-box-in"><div className="md-hurdle-box-title">01 예측 정보 입력</div><div className="md-hurdle-box-sub">예측 시점 기준 입력값 구성</div></div>
        <span className="md-hurdle-arrow">→</span>
        <div className="md-hurdle-box md-hurdle-box-cls"><div className="md-hurdle-box-title">02 판매 발생 여부 분류</div><div className="md-hurdle-box-sub">판매량 0 / 1 이상을 분류해 발생확률 예측</div></div>
        <span className="md-hurdle-arrow">→</span>
        <div className="md-hurdle-box md-hurdle-box-reg"><div className="md-hurdle-box-title">03 판매 발생 시 수량 회귀</div><div className="md-hurdle-box-sub">판매가 발생한 행만 log1p 판매량 학습</div></div>
        <span className="md-hurdle-arrow">→</span>
        <div className="md-hurdle-box md-hurdle-box-out"><div className="md-hurdle-box-title">04 최종 수요 예측</div><div className="md-hurdle-box-sub">판매 발생확률 × 발생 시 예상 판매량</div></div>
      </div>
      <div className="md-hurdle-result-strip">
        <div><span>기존 LightGBM</span><strong>WAPE 59.50% · Bias -31.63%</strong></div>
        <span className="md-hurdle-result-arrow">→</span>
        <div className="is-selected"><span>Hurdle + log1p</span><strong>WAPE 60.19% · Bias -18.80%</strong></div>
      </div>
      <div className="md-hurdle-conclusion">WAPE는 유사하게 유지하면서 과소예측을 Bias 허용범위 내로 개선</div>
      {open && <BiasAlternativesModal onClose={() => setOpen(false)} />}
    </div>
  );
}

function BiasAlternativesModal({ onClose }) {
  const methods = [
    ['기존 LGBM · log1p', '59.50%', '-31.63%', 'Bias 기준 미통과'],
    ['Raw-scale', '67.84%', '+3.45%', 'WAPE 악화'],
    ['Hurdle + Raw', '65.97%', '+2.66%', 'WAPE 악화'],
    ['Hurdle + log1p', '60.19%', '-18.80%', '채택'],
    ['Tweedie', '63.27~64.71%', '-2.05~+1.05%', 'WAPE 열위'],
  ];
  const alternatives = [
    { number: '1', title: '기존 문제 — log1p LightGBM', rows: [['확인 결과', 'WAPE 59.50%로 오차는 낮았지만 Bias -31.63%로 과소예측이 큼'], ['원인 후보', 'target의 log1p → expm1 변환 영향']] },
    { number: '2', title: 'log1p 제거 — Raw-scale', rows: [['실험', '동일 LightGBM 설정에서 target 변환만 제거'], ['목적', '음의 Bias가 log1p와 관련 있는지 확인'], ['결과', 'Bias +3.45% · WAPE 67.84%'], ['판단', 'Bias는 해결됐지만 오차 증가가 커 제외']] },
    { number: '3', title: '0수요 분리 — Hurdle + Raw', rows: [['실험', '판매 발생 여부 분류 → 발생 시 판매량 회귀 구조로 분리하고 회귀는 raw-scale로 학습'], ['목적', '0수요와 양수 수요를 분리하는 것 자체가 Bias 개선에 효과가 있는지 확인'], ['결과', 'Bias +2.66% · WAPE 65.97%'], ['판단', 'Bias는 개선됐지만 WAPE 손실이 커 제외']] },
    { number: '4', title: 'Hurdle + log1p', selected: true, rows: [['실험', 'Hurdle 구조를 유지하고 양수 수요 회귀에 log1p 적용'], ['목적', 'Hurdle의 Bias 개선 효과와 log1p의 낮은 WAPE를 함께 확보'], ['결과', 'WAPE 60.19% · Bias -18.80%'], ['판단', 'Bias ±20% 통과 + 기존 대비 WAPE 증가 0.69%p에 그쳐 채택']] },
    { number: '5', title: '분포 기반 대안 — Tweedie', rows: [['정의', '0이 많고 양수값이 오른쪽으로 치우친 비음수 데이터를 하나의 회귀모델에서 처리'], ['실험', 'LightGBM 목적함수를 Tweedie로 변경하고 variance_power=1.1 / 1.2 / 1.5 / 1.8 비교'], ['목적', '모델을 둘로 나누지 않고 0수요와 양수 수요를 함께 처리할 수 있는지 확인'], ['결과', 'Bias -2.05~+1.05% · WAPE 63.27~64.71%'], ['판단', 'Bias는 우수하지만 Hurdle + log1p보다 WAPE가 모두 높아 제외']] },
  ];
  return <div className="md-modal-backdrop" role="presentation" onClick={(e) => { e.stopPropagation(); onClose(); }}>
    <section className="md-modal md-bias-modal" role="dialog" aria-modal="true" aria-labelledby="bias-alternatives-title" onClick={(e) => e.stopPropagation()}>
      <header className="md-modal-hd"><div><h2 id="bias-alternatives-title">Bias 개선 대안 검증</h2><p>기존 LightGBM의 과소예측 원인을 확인하고, Bias를 줄이면서 WAPE 손실을 최소화하는 방법을 비교함</p></div><button type="button" className="md-modal-close" aria-label="닫기" onClick={onClose}>×</button></header>
      <div className="md-modal-body md-bias-modal-body">
        <div className="md-bias-alternative-grid">{alternatives.map((item) => <section key={item.number} className={item.selected ? 'is-selected' : ''}><h3><span>{item.number}</span>{item.title}{item.selected && <b>채택</b>}</h3><dl>{item.rows.map(([label, text]) => <div key={label}><dt>{label}</dt><dd>{text}</dd></div>)}</dl></section>)}</div>
        <section className="md-bias-comparison"><h3>결과 비교</h3><table><thead><tr><th>방법</th><th>WAPE</th><th>Bias</th><th>판단</th></tr></thead><tbody>{methods.map((row) => <tr key={row[0]} className={row[0] === 'Hurdle + log1p' ? 'is-selected' : ''}>{row.map((cell) => <td key={cell}>{cell}</td>)}</tr>)}</tbody></table></section>
        <p className="md-bias-conclusion"><b>결론</b><span><code>log1p</code> 제거만으로 Bias는 개선됐지만 WAPE가 크게 악화됨. 최종적으로 0수요와 양수 수요를 분리하는 Hurdle 구조에 <code>log1p</code> 회귀를 결합했을 때 Bias 기준을 통과하면서 WAPE 손실을 가장 작게 억제하여 채택함.</span></p>
      </div>
    </section>
  </div>;
}

function HurdleHpoTab() {
  const [model, setModel] = useState('H-LGBM');
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;
    setDetail(null); setError(null);
    api.getMlDlModelDetail({ model, horizon: 'h1' })
      .then((d) => { if (!ignore) setDetail(d); })
      .catch((e) => { if (!ignore) setError(e.message); });
    return () => { ignore = true; };
  }, [model]);

  const hpoRow = detail?.summary.find((r) => r.parameter_type === 'hpo');
  const fixedRow = detail?.summary.find((r) => r.parameter_type === 'fixed');

  return (
    <div className="md-hpo-tab">
      <div className="md-seg">
        {['H-RF', 'H-LGBM'].map((m) => (
          <button key={m} className={`md-seg-btn ${model === m ? 'active' : ''}`} onClick={() => setModel(m)}>{m}</button>
        ))}
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !detail && <div className="az-hint">불러오는 중...</div>}
      {!error && detail && (
        <div className="md-hpo-body">
          {hpoRow && (
            <div className="md-detail-row-line"><span className="md-detail-row-key">튜닝 파라미터</span><span>{hpoRow.parameter}</span></div>
          )}
          {hpoRow && <KVGrid text={hpoRow.candidates_or_rule} />}
          <div className="md-detail-row-line"><span className="md-detail-row-key">탐색 방식</span><span>사전에 정의한 파라미터 조합을 전수 검증(Trial 개수는 모델마다 다르며 우열 비교에 쓰지 않음)</span></div>
          {hpoRow && (
            <div className="md-detail-row-line"><span className="md-detail-row-key">1주·2주·4주 후 최종값</span><span>{hpoRow.selected_or_fixed}</span></div>
          )}
          <div className="md-detail-row-line"><span className="md-detail-row-key">threshold</span><span>별도 탐색 없음 — 발생 확률 × 조건부 회귀값 결합</span></div>
          {fixedRow && (
            <div className="md-detail-row-line"><span className="md-detail-row-key">고정값</span><span>{fixedRow.selected_or_fixed}</span></div>
          )}
          <div className="md-detail-row-line"><span className="md-detail-row-key">2023 CV</span><span>A센터, 4-fold expanding window(2023 Q1~Q4를 순차적으로 검증구간 사용)</span></div>
          <div className="md-detail-row-line"><span className="md-detail-row-key">선정 순서</span><span>① |4개 검증구간 통합 Bias| ≤ 20% 통과 → ② 통과 trial 중 4개 검증구간 통합 WAPE 최소 → ③ 1%p 이내 near-tie면 최악 검증구간 WAPE 최소</span></div>
          {hpoRow && <div className="md-detail-row-line"><span className="md-detail-row-key">선정 기준</span><span>{hpoRow.selection_criterion}</span></div>}
        </div>
      )}
    </div>
  );
}

function HurdleExperimentsTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getMlDlImprovementExperiments().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="az-hint az-hint-error">{error}</div>;
  if (!data) return <div className="az-hint">불러오는 중...</div>;

  return (
    <div className="md-experiments-tab">
      {data.entries.map((entry) => (
        <div className="md-experiment-block" key={entry.id}>
          <div className="md-experiment-title">{entry.title}</div>
          <div className="md-detail-caption">{entry.hypothesis} · {entry.changed}</div>
          <div className="md-detail-caption md-eval-note">{entry.eval_note}</div>
          <table className="md-table">
            <thead><tr><th>variant</th><th>WAPE(%)</th><th>Bias(%)</th><th /></tr></thead>
            <tbody>
              {entry.variants.map((v) => (
                <tr key={v.key}>
                  <td>{v.label}</td>
                  <td>{fmtMetric(v.wape)}</td>
                  <td>{fmtMetric(v.bias)}</td>
                  <td>{v.adopted && <span className="md-final-badge">채택</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="md-detail-caption">{entry.conclusion}</p>
          <p className="md-source-note">근거: {entry.source_paths.join(', ')}</p>
        </div>
      ))}
      {data.narrative_notes.map((n) => (
        <div className="md-experiment-block" key={n.id}>
          <div className="md-experiment-title">{n.title}</div>
          <p className="md-detail-caption">{n.body}</p>
          <p className="md-source-note">근거: {n.source_paths.join(', ')}</p>
        </div>
      ))}
    </div>
  );
}

const BEFORE_AFTER_PAIRS = [
  { base: 'RF', hurdle: 'Hurdle-RF' },
  { base: 'LightGBM', hurdle: 'Hurdle-LightGBM' },
];
function BeforeAfterCard({ summary, error }) {
  const biasValues = summary ? BEFORE_AFTER_PAIRS.flatMap((pair) => HORIZON_KEYS.flatMap((h) => [
    summary.entries.find((e) => e.model === pair.base && e.horizon === h)?.bias,
    summary.entries.find((e) => e.model === pair.hurdle && e.horizon === h)?.bias,
  ])).filter(Number.isFinite) : [];
  const biasBounds = biasValues.length ? {
    min: Math.floor((Math.min(...biasValues, -20) - 5) / 10) * 10,
    max: Math.ceil((Math.max(...biasValues, 20) + 5) / 10) * 10,
  } : { min: -70, max: 30 };
  return <div className="az-card">
    <div className="az-card-hd"><h3>4. Hurdle 적용 전후 Bias 개선 <span className="md-dev-badge">A센터 개발검증</span></h3><p>기존 모델 대비 Hurdle 적용 결과</p></div>
    {error && <div className="az-hint">{error}</div>}
    {!error && !summary && <div className="az-hint">불러오는 중...</div>}
    {summary && <><div className="md-bias-visual-key"><span><i className="is-before" />적용 전</span><span><i className="is-after" />Hurdle 적용 후</span><span><i className="is-band" />Bias 허용구간 -20%~+20%</span></div>
      <div className="md-bias-pair-grid">{BEFORE_AFTER_PAIRS.map((pair) => <section key={pair.base}><h4>{pair.base === 'RF' ? 'Random Forest → Hurdle-Random Forest' : 'LightGBM → Hurdle-LightGBM'}</h4><EChart fill option={buildBeforeAfterOption(summary, pair, biasBounds)} /></section>)}</div>
      <table className="md-wape-delta"><caption>Hurdle 적용 후 WAPE 변화</caption><thead><tr><th>적용 모델</th>{HORIZONS.map((h) => <th key={h.value}>{h.label}</th>)}</tr></thead><tbody>{BEFORE_AFTER_PAIRS.map((pair) => <tr key={pair.base} className={pair.base === 'LightGBM' ? 'is-lightgbm' : ''}><th>{MLDL_MODEL_LABELS[pair.base]} → {MLDL_MODEL_LABELS[pair.hurdle]}</th>{HORIZON_KEYS.map((h) => {
        const before = summary.entries.find((e) => e.model === pair.base && e.horizon === h)?.wape;
        const after = summary.entries.find((e) => e.model === pair.hurdle && e.horizon === h)?.wape;
        const delta = Number.isFinite(before) && Number.isFinite(after) ? after - before : null;
        return <td key={h} title={`${fmtMetric(before)}% → ${fmtMetric(after)}%`}>{delta === null ? '—' : `${delta > 0 ? '+' : ''}${fmtMetric(delta)}%p`}</td>;
      })}</tr>)}</tbody></table></>}
  </div>;
}

function buildBeforeAfterOption(summary, pair, bounds) {
  const rows = HORIZON_KEYS.map((h) => ({
    label: horizonLabel(h),
    before: summary.entries.find((e) => e.model === pair.base && e.horizon === h)?.bias,
    after: summary.entries.find((e) => e.model === pair.hurdle && e.horizon === h)?.bias,
  }));
  return {
    grid: { left: pair.base === 'RF' ? 38 : 28, right: 8, top: 8, bottom: 25 },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis', formatter: (params) => {
      const row = rows[params[0].dataIndex];
      const improvement = Number.isFinite(row.before) && Number.isFinite(row.after) ? Math.abs(row.before) - Math.abs(row.after) : null;
      return `${pair.base === 'RF' ? 'RF → H-RF' : 'LGBM → H-LGBM'} · ${row.label}<br/>적용 전 Bias: ${fmtMetric(row.before)}%<br/>적용 후 Bias: ${fmtMetric(row.after)}%<br/>Bias 개선폭: ${fmtMetric(improvement)}%p`;
    } },
    xAxis: { type: 'category', data: rows.map((p) => p.label), axisLabel: { fontSize: 9.5, fontWeight: 650, interval: 0 } },
    yAxis: { type: 'value', min: bounds.min, max: bounds.max, interval: 20, axisLabel: { show: pair.base === 'RF', formatter: (v) => Number(v).toFixed(0), fontSize: 9 }, splitLine: { lineStyle: { color: '#edf2f7' } } },
    series: [
      { name: '적용 전', type: 'bar', barMaxWidth: 17, itemStyle: { color: '#f9a8d4' }, data: rows.map((p) => p.before ?? null), label: { show: true, position: 'bottom', distance: 3, color: '#be185d', fontSize: 9, formatter: (p) => fmtMetric(p.value) },
        markArea: { silent: true, itemStyle: { color: 'rgba(22,163,74,.10)' }, label: { show: false }, data: [[{ yAxis: -20 }, { yAxis: 20 }]] }, markLine: { silent: true, symbol: 'none', label: { show: false }, data: [{ yAxis: -20, lineStyle: { color: '#86efac', width: 1 } }, { yAxis: 0, lineStyle: { color: '#16a34a', width: 1.4 } }, { yAxis: 20, lineStyle: { color: '#86efac', width: 1 } }] } },
      { name: 'Hurdle 적용 후', type: 'bar', barMaxWidth: 17, itemStyle: { color: '#2563eb' }, data: rows.map((p) => p.after ?? null), label: { show: true, position: 'top', distance: 3, color: '#1d4ed8', fontSize: 9, fontWeight: 700, formatter: (p) => fmtMetric(p.value) } },
    ],
  };
}

const HURDLE_METRICS = [
  { value: 'wape', label: 'WAPE' },
  { value: 'bias', label: 'Bias' },
  { value: 'mae', label: 'MAE' },
  { value: 'rmse', label: 'RMSE' },
];

function hurdleMetricAvailable(summary, metric) {
  if (!summary) return false;
  if (metric === 'wape' || metric === 'bias') return true;
  const statusKey = metric === 'mae' ? 'mae_status' : 'rmse_status';
  return summary.entries
    .filter((e) => e.model === 'Hurdle-RF' || e.model === 'Hurdle-LightGBM')
    .every((e) => e[statusKey] === 'ok');
}

function HurdleCompareCard({ summary, error }) {
  const rows = HORIZON_KEYS.map((h) => ({ h,
    rf: summary?.entries.find((e) => e.model === 'Hurdle-RF' && e.horizon === h),
    lg: summary?.entries.find((e) => e.model === FINAL_MODEL && e.horizon === h),
  }));
  const bothPass = rows.every((r) => r.rf?.bias_pass && r.lg?.bias_pass);
  const winsAll = rows.every((r) => Number.isFinite(r.rf?.wape) && Number.isFinite(r.lg?.wape) && r.lg.wape < r.rf.wape);
  const values = rows.flatMap((r) => [r.rf?.wape, r.lg?.wape]).filter(Number.isFinite);
  const axisMin = values.length ? Math.max(0, Math.floor((Math.min(...values) - 2) / 2) * 2) : 0;
  const axisMax = values.length ? Math.ceil((Math.max(...values) + 2) / 2) * 2 : 10;
  return <div className="az-card">
    <div className="az-card-hd"><h3>5. 후속 Hurdle 후보 성능 비교 <span className="md-dev-badge">A센터 개발검증</span></h3><p>H-RF vs H-LGBM</p></div>
    {error && <div className="az-hint">{error}</div>}
    {!error && !summary && <div className="az-hint">불러오는 중...</div>}
    {summary && <>
      <EChart fill option={{
        grid: { left: 68, right: 28, top: 32, bottom: 24 },
        legend: { top: 0, data: ['H-RF', 'H-LGBM'] },
        tooltip: { ...TOOLTIP_BASE, trigger: 'axis', formatter: (params) => {
          const row = rows[params[0].dataIndex];
          return `${horizonLabel(row.h)}<br/>H-RF WAPE: ${fmtMetric(row.rf?.wape)}%<br/>H-LGBM WAPE: ${fmtMetric(row.lg?.wape)}%`;
        } },
        xAxis: { type: 'value', min: axisMin, max: axisMax, interval: 2, name: 'WAPE (%)', axisLabel: { formatter: (v) => Number(v).toFixed(0) }, splitLine: { lineStyle: { color: '#e9eef5' } } },
        yAxis: { type: 'category', inverse: true, data: rows.map((r) => horizonLabel(r.h)) },
        graphic: [{ type: 'text', left: 48, bottom: 13, style: { text: `//  축 시작 ${axisMin}%`, fill: '#64748b', font: '10px sans-serif' } }],
        series: ['rf', 'lg'].map((key, i) => ({
          name: i ? 'H-LGBM' : 'H-RF', type: 'bar', barMaxWidth: 28,
          itemStyle: { color: i ? '#2563eb' : '#93c5fd' }, data: rows.map((r) => r[key]?.wape ?? null),
          label: { show: true, position: 'right', color: '#1e40af', fontSize: 11, formatter: (params) => fmtMetric(params.value) },
        })),
      }} />
      <div className="md-selection-result">{bothPass && winsAll ? '1주 · 2주 · 4주 후 모두 H-LGBM WAPE가 더 낮음' : 'Bias 기준 통과 후보의 WAPE 비교'}</div>
    </>}
    <div className="md-caption">확대 축 · 막대가 짧을수록 WAPE 우수 · 막대 위 숫자는 실제 WAPE</div>
  </div>;
}

function buildHurdleCompareOption(summary, metric, horizon) {
  const models = ['Hurdle-RF', FINAL_MODEL];
  return {
    grid: { left: 48, right: 16, top: 30, bottom: 28 },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis', valueFormatter: fmtMetric },
    xAxis: { type: 'category', data: ['H-RF', 'H-LGBM'] },
    yAxis: { type: 'value', name: HURDLE_METRICS.find((m) => m.value === metric).label, axisLabel: { formatter: (v) => Number(v).toFixed(0) } },
    series: [{ type: 'bar', barMaxWidth: 56, data: models.map((model, i) => ({ value: summary.entries.find((e) => e.model === model && e.horizon === horizon)?.[metric] ?? null, itemStyle: { color: i === 0 ? '#ea580c' : FINAL_COLOR } })), label: { show: true, position: 'top', formatter: (p) => fmtMetric(p.value), fontSize: 12 }, ...(metric === 'bias' ? { markArea: { silent: true, itemStyle: { color: 'rgba(16,185,129,.06)' }, data: [[{ yAxis: -20 }, { yAxis: 20 }]] } } : {}) }],
  };
}

function useModelParamDetails(model) {
  const [details, setDetails] = useState(null);
  useEffect(() => {
    let ignore = false;
    setDetails(null);
    Promise.all(HORIZON_KEYS.map((h) =>
      api.getMlDlModelDetail({ model: MLDL_MODEL_LABELS[model], horizon: h })
        .then((d) => [h, d]).catch(() => [h, null]),
    )).then((pairs) => { if (!ignore) setDetails(Object.fromEntries(pairs)); });
    return () => { ignore = true; };
  }, [model]);
  return details;
}

function FinalModelCard({ summary, error, selectedModel }) {
  const [detailOpen, setDetailOpen] = useState(false);
  const paramDetails = useModelParamDetails(selectedModel);
  const isFinal = selectedModel === FINAL_MODEL;
  const config = ML_DL_MODEL_DETAIL[selectedModel];
  const varCount = config.variableGroups('h1').reduce((s, g) => s + g.items.length, 0);
  const chips = isFinal ? [`입력 ${varCount}개`, '분류 + 회귀', '16개 조합 비교'] : [`입력 ${varCount}개`, config.track === 'dl' ? '시계열 신경망' : config.track === 'hurdle' ? 'Hurdle(분류 + 회귀)' : '머신러닝', '2023 CV 비교'];

  return <div className="az-card md-clickable-summary" role="button" tabIndex={0} aria-haspopup="dialog" onClick={() => setDetailOpen(true)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); setDetailOpen(true); } }}>
    <div className="az-card-hd"><h3>6. {isFinal ? 'ML/DL 대표모델 — Hurdle-LightGBM' : '선택 모델 소개'}</h3><p>A센터 2023 개발검증 결과</p></div>
    <div className="md-final-title-row">
      <strong className="az-detail-title">{MLDL_MODEL_LABELS[selectedModel]}</strong>
      {isFinal ? <span className="md-final-badge">최종 선정</span> : <span className="md-ref-badge">최종 미선정</span>}
    </div>
    <p className="md-detail-caption">{isFinal ? '수요 발생 여부와 발생량을 분리해 예측하는 2단계 모델' : config.structure.purpose}</p>
    <div className="md-summary-chips">{chips.map((chip) => <span key={chip}>{chip}</span>)}</div>
    {isFinal && <div className="md-final-process">판매 발생 확률 → 발생 시 판매량 → 최종 결합</div>}
    {error && <div className="az-hint">{error}</div>}
    {!error && !summary && <div className="az-hint">불러오는 중...</div>}
    {summary && <div className="md-summary-performance">{HORIZON_KEYS.map((h) => { const entry = summary.entries.find((e) => e.model === selectedModel && e.horizon === h); return <div key={h}><b>{horizonLabel(h)}</b>{entry ? <><span>WAPE <strong>{fmtMetric(entry.wape)}%</strong></span><span>Bias <strong>{fmtMetric(entry.bias)}%</strong></span></> : <span>데이터 없음</span>}</div>; })}</div>}
    {detailOpen && <MlDlDetailModal model={selectedModel} summary={summary} paramDetails={paramDetails} onClose={() => setDetailOpen(false)} />}
  </div>;
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

function MlDlDetailModal({ model, summary, paramDetails, onClose }) {
  const config = ML_DL_MODEL_DETAIL[model];
  const [varHorizon, setVarHorizon] = useState('h1');
  const [performanceView, setPerformanceView] = useState('cv');
  const [holdout2024, setHoldout2024] = useState(null);
  const isFinalModel = model === FINAL_MODEL;

  useEffect(() => {
    if (!isFinalModel) return undefined;
    let ignore = false;
    Promise.all(['WAPE', 'Bias', 'MAE', 'RMSE'].map((metric) =>
      api.getFinalKpi({ center: 'ALL', horizon: 'ALL', metric, scope: 'full_common' })
        .then((d) => [metric, d.entries]).catch(() => [metric, null]),
    )).then((pairs) => { if (!ignore) setHoldout2024(Object.fromEntries(pairs)); });
    return () => { ignore = true; };
  }, [isFinalModel]);

  const isHurdle = config.track === 'hurdle';
  const groups = config.variableGroups(varHorizon);
  const anyBiasPass = HORIZON_KEYS.some((h) => summary?.entries.find((e) => e.model === model && e.horizon === h)?.bias_pass);

  const sections = [
    {
      id: 'overview', heading: '개요',
      body: <>
        <div className="mdl-kv-row"><span>사용 구현</span><span>{config.structure.library}</span></div>
        <div className="mdl-kv-row"><span>모델 단위</span><span>{config.structure.unit}</span></div>
        <div className="mdl-kv-row"><span>사용 목적</span><span>{config.structure.purpose}</span></div>
        <div className="mdl-flow">{(isHurdle ? ['30개 Feature', '판매발생 분류 + 양수 판매량 회귀', '확률 × 판매량', '최종예측'] : config.structure.flow).map((step, i) => <span key={step}>{i > 0 && <span className="mdl-flow-arrow">→ </span>}{step}</span>)}</div>
      </>,
    },
    {
      id: 'variables', heading: '사용 변수',
      body: <>
        <div className="mdl-inline-seg">
          <span className="mdl-note">공휴일 변수는 예측시점별로 컬럼명이 달라집니다:</span>
          <div className="md-seg">{HORIZONS.map((h) => (
            <button key={h.value} className={`md-seg-btn ${varHorizon === h.value ? 'active' : ''}`} onClick={() => setVarHorizon(h.value)}>{h.label}</button>
          ))}</div>
        </div>
        {isHurdle && <div className="mdl-note">분류기와 회귀기는 동일한 변수 목록을 입력으로 사용합니다(별도 축소 없음).</div>}
        <div className="mdl-variable-groups">{groups.map((g) => (
          <details className="mdl-variable-group" key={g.group}>
            <summary>{g.group}<span>{g.items.length}개</span></summary>
            <table className="mdl-table">
              <thead><tr><th>컬럼명</th><th>컬럼 설명</th><th>활용 목적</th></tr></thead>
              <tbody>{g.items.map((c) => <tr key={c.col}><td><code>{c.col}</code></td><td>{c.meaning}</td><td>{c.purpose}</td></tr>)}</tbody>
            </table>
          </details>
        ))}</div>
      </>,
    },
    {
      id: 'preprocessing', heading: '전처리',
      body: <>
        <div className="mdl-note">공통 데이터 전처리는 제외하고, 이 모델에 입력하기 직전에만 수행한 처리입니다.</div>
        <table className="mdl-table">
          <thead><tr><th>처리 대상</th><th>처리 방법</th><th>적용 이유</th></tr></thead>
          <tbody>{config.preprocessing.map((p) => <tr key={p.target}><td>{p.target}</td><td>{p.method}</td><td>{p.reason}</td></tr>)}</tbody>
        </table>
      </>,
    },
    {
      id: 'parameters', heading: '파라미터',
      body: <>
        <div className="mdl-source-cards"><div className="mdl-source-card"><b>선행연구</b><strong>{config.paramRationale.priorResearch.source}</strong><span>{config.paramRationale.priorResearch.reason}</span></div><div className="mdl-source-card"><b>공식문서</b><strong>{config.paramRationale.officialDocs.source}</strong><span>{config.paramRationale.officialDocs.reason}</span></div></div>
        {paramDetails ? <table className="mdl-table"><thead><tr><th>파라미터</th><th>목적</th><th>선행연구 범위</th><th>공식 기준</th><th>실제 적용</th></tr></thead>
          <tbody>
            {(() => {
              const rows = paramDetails.h1?.summary || [];
              const hpo = rows.find((r) => r.parameter_type === 'hpo');
              const fixed = rows.find((r) => r.parameter_type === 'fixed');
              return <>
                {hpo && <tr><td><code>{hpo.parameter}</code></td><td>후보 조합 탐색</td><td>{config.paramRationale.priorResearch.rows.map((r) => r.range).join(' · ') || '—'}</td><td>{config.paramRationale.officialDocs.rows.map((r) => r.range).join(' · ') || '—'}</td><td><ModeTag mode="search" /> {hpo.candidates_or_rule}</td></tr>}
                {fixed && <tr><td><code>{fixed.parameter}</code></td><td>공통 학습 설정</td><td>—</td><td>—</td><td><ModeTag mode="fixed" /> {fixed.selected_or_fixed}</td></tr>}
              </>;
            })()}
          </tbody>
        </table> : <div className="mdl-loading">불러오는 중...</div>}
        <div className="mdl-sub-hd">최종 파라미터</div>
        {!paramDetails ? <div className="mdl-loading">불러오는 중...</div> : anyBiasPass ? (isFinalModel ? <><table className="mdl-table"><thead><tr><th>예측시점</th><th>분류기</th><th>회귀기</th></tr></thead><tbody><tr><td>1주 후</td><td><code>num_leaves=31 · min_child_samples=100</code></td><td><code>num_leaves=31 · min_child_samples=100</code></td></tr><tr><td>2주 후</td><td><code>num_leaves=31 · min_child_samples=1000</code></td><td><code>num_leaves=31 · min_child_samples=100</code></td></tr><tr><td>4주 후</td><td><code>num_leaves=31 · min_child_samples=1000</code></td><td><code>num_leaves=31 · min_child_samples=100</code></td></tr></tbody></table><p className="mdl-note"><code>learning_rate=0.1</code> · <code>n_estimators=100</code> · regressor target=<code>log1p</code> · soft 결합</p></> : <table className="mdl-table"><thead><tr><th>예측시점</th><th>최종값</th><th>구분</th></tr></thead><tbody>{HORIZON_KEYS.map((h) => { const hpo = paramDetails[h]?.summary.find((row) => row.parameter_type === 'hpo'); return <tr key={h}><td>{horizonLabel(h)}</td><td>{hpo?.selected_or_fixed || '—'}</td><td><ModeTag mode="search" /></td></tr>; })}</tbody></table>) : <div className="mdl-final-status is-none">Bias 기준 통과 후보 없음 · 최종 선정 파라미터 없음</div>}
        <details className="mdl-compact-details"><summary>선정 근거 상세 보기</summary><div className="mdl-lead">{config.hpoNarrative}</div><div className="mdl-lead">{config.fixedNarrative}</div>{config.lookbackNote && <div className="mdl-lead">{config.lookbackNote}</div>}{config.extraNotes.map((note) => <div className="mdl-note" key={note}>{note}</div>)}</details>
      </>,
    },
    {
      id: 'performance', heading: '성능',
      body: <>
        <p className="mdl-note">Bias ±20% 통과 → 통과 후보 중 WAPE 최소 → 근접 시 worst-fold WAPE 확인</p>
        <div className="md-seg"><button type="button" className={`md-seg-btn ${performanceView === 'cv' ? 'active' : ''}`} onClick={() => setPerformanceView('cv')}>2023 모델선정 검증</button><button type="button" disabled={!isFinalModel} className={`md-seg-btn ${performanceView === 'holdout' ? 'active' : ''}`} onClick={() => setPerformanceView('holdout')}>2024 Final Holdout</button></div>
        {performanceView === 'cv' && <><div className="mdl-perf-hint">A센터 4-fold expanding CV · {anyBiasPass ? '기준 통과 selected trial' : '참고용 최저 WAPE trial'}</div><table className="mdl-table mdl-performance-table">
          <thead><tr><th>예측시점</th><th>WAPE(%)</th><th>Bias(%)</th><th>MAE</th><th>RMSE</th></tr></thead>
          <tbody>{HORIZON_KEYS.map((h) => {
            const e = summary?.entries.find((x) => x.model === model && x.horizon === h);
            return <tr key={h}>
              <td>{horizonLabel(h)}</td>
              <td>{e ? fmtMetric(e.wape) : '-'}</td><td>{e ? fmtMetric(e.bias) : '-'}</td>
              <td className={e?.mae_status === 'ok' ? '' : 'mdl-perf-unavailable'}>{e?.mae_status === 'ok' ? fmtMetric(e.mae) : 'artifact에 없음'}</td>
              <td className={e?.rmse_status === 'ok' ? '' : 'mdl-perf-unavailable'}>{e?.rmse_status === 'ok' ? fmtMetric(e.rmse) : 'artifact에 없음'}</td>
            </tr>;
          })}</tbody>
        </table></>}
        {performanceView === 'holdout' && (isFinalModel ? (
          holdout2024 ? <>
            <div className="mdl-perf-hint">전체 센터 · A+B 통합 재학습 모델 · 2023 CV와 직접 비교하지 않음</div>
            <table className="mdl-table mdl-performance-table">
              <thead><tr><th>예측시점</th><th>WAPE(%)</th><th>Bias(%)</th><th>MAE</th><th>RMSE</th></tr></thead>
              <tbody>{HORIZON_KEYS.map((h) => {
                const pick = (metric) => holdout2024[metric]?.find((x) => x.horizon === h)?.ml_value;
                return <tr key={h}>
                  <td>{horizonLabel(h)}</td>
                  <td>{fmtMetric(pick('WAPE'))}</td><td>{fmtMetric(pick('Bias'))}</td>
                  <td>{fmtMetric(pick('MAE'))}</td><td>{fmtMetric(pick('RMSE'))}</td>
                </tr>;
              })}</tbody>
            </table>
          </> : <div className="mdl-loading">불러오는 중...</div>
        ) : <div className="mdl-final-status is-none">2024 Holdout 없음 — 최종 재학습 모델만 제공됩니다.</div>)}
      </>,
    },
  ];

  const statusBadge = isFinalModel
    ? { tone: 'final', text: '최종 선정 모델' }
    : anyBiasPass ? { tone: 'normal', text: 'Bias 기준 통과 · 최종 미선정' } : { tone: 'excluded', text: 'Bias 기준 미통과' };

  return <ModelDetailModal title={MLDL_MODEL_LABELS[model]} typeLabel={modelTypeLabel(model)} statusBadge={statusBadge} sections={sections} onClose={onClose} />;
}
