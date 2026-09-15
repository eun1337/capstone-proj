import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { api } from '../../api/client.js';
import EChart from '../../components/charts/EChart.jsx';
import './analysis.css';
import './MlDlAnalysis.css';

const HORIZONS = [
  { value: 'h1', label: '1주 후' },
  { value: 'h2', label: '2주 후' },
  { value: 'h4', label: '4주 후' },
];
const MODELS = [
  { value: 'RF', label: 'RF' },
  { value: 'LGBM', label: 'LGBM' },
  { value: 'LSTM', label: 'LSTM' },
  { value: 'TFT', label: 'TFT' },
  { value: 'Informer', label: 'Informer' },
  { value: 'H-RF', label: 'H-RF' },
  { value: 'H-LGBM', label: 'H-LGBM' },
];
const TRIAL_FILTERS = [
  { value: 'ALL', label: '전체 Trial' },
  { value: 'PASS', label: 'Bias 기준 통과' },
  { value: 'SELECTED', label: '최종 후보' },
];

const MLDL_MODEL_LABELS = {
  RF: 'RF', LightGBM: 'LGBM', LSTM: 'LSTM', TFT: 'TFT', Informer: 'Informer',
  'Hurdle-RF': 'H-RF', 'Hurdle-LightGBM': 'H-LGBM',
};
const horizonLabel = (h) => HORIZONS.find((item) => item.value === h)?.label || '—';
const LABEL_TO_MODEL = Object.fromEntries(Object.entries(MLDL_MODEL_LABELS).map(([k, v]) => [v, k]));
const MODEL_COLORS = {
  RF: '#3b82f6', LightGBM: '#0d9488', LSTM: '#7c9db5', TFT: '#527f99', Informer: '#64748b',
  'Hurdle-RF': '#14b8a6', 'Hurdle-LightGBM': '#2563eb',
};
const HURDLE_MODELS = new Set(['Hurdle-RF', 'Hurdle-LightGBM']);
const HURDLE_GROUP_BOUNDARY = 4.5;

const ML_BASE_FEATURES = [
  '입수', 'KAN_대분류', 'KAN_중분류', 'KAN_소분류', 'ISO_주차', '평균온도', '총강수량',
  'existed_before_regime', 'qty_log1p', '월', '분기', 'is_warmup', 'coldstart_flag',
  'adi_expanding_filled', 'cv2_expanding_filled', '강수량_호우_count', 'covid_flag',
  'center_is_B', 'temp_x_precip', 'center_temp_inter', 'weeks_since_last_active_filled',
  'ccsi_lag_m1', 'cpi_y1_prev', 'cpi_y2_prev_yoy',
];
const ML_DEMAND_SUMMARY_FEATURES = ['qty_lag1_filled_log1p', 'qty_rollmean_4_filled_log1p', 'qty_rollstd_4_filled_log1p'];

const DL_STATIC_CATEGORICAL = ['KAN_대분류', 'KAN_중분류', 'KAN_소분류'];
const DL_STATIC_CONTINUOUS = ['입수', 'existed_before_regime', 'center_is_B'];
const DL_TIME_VARYING_KNOWN = ['ISO_주차', '월', '분기', 'covid_flag'];
const DL_TIME_VARYING_OBSERVED = [
  '평균온도', '총강수량', 'qty_log1p', 'is_warmup', 'coldstart_flag', 'adi_expanding_filled',
  'cv2_expanding_filled', '강수량_호우_count', 'temp_x_precip', 'center_temp_inter',
  'weeks_since_last_active_filled', 'ccsi_lag_m1', 'cpi_y1_prev', 'cpi_y2_prev_yoy',
];

const HOLIDAY_FEATURES_BY_HORIZON = {
  h1: ['target_h1_공휴일_W0', 'target_h1_공휴일_W-1', 'target_h1_공휴일_W+1'],
  h2: ['target_h2_공휴일_W0', 'target_h2_공휴일_W-1', 'target_h2_공휴일_W+1'],
  h4: ['target_h4_공휴일_W0', 'target_h4_공휴일_W-1', 'target_h4_공휴일_W+1'],
};
const TARGET_COL_BY_HORIZON = { h1: 'target_h1', h2: 'target_h2', h4: 'target_h4' };
const DL_MODELS = new Set(['LSTM', 'TFT', 'Informer']);
const LGBM_FAMILY_MODELS = new Set(['LightGBM', 'Hurdle-LightGBM']);

const PREPROCESSING_TEXT = {
  rf: '결측 5종(adi_expanding_filled, cv2_expanding_filled, qty_lag1_filled_log1p, qty_rollmean_4_filled_log1p, qty_rollstd_4_filled_log1p)은 train 기준 KAN 소→중→대분류→전체 순 계층형 median으로 대체. KAN 대/중/소분류는 OneHotEncoder로 인코딩(train에서 학습한 카테고리만 사용).',
  lgbm: '구조적 결측 5종(adi_expanding_filled, cv2_expanding_filled, qty_lag1_filled_log1p, qty_rollmean_4_filled_log1p, qty_rollstd_4_filled_log1p)은 대체하지 않고 LightGBM native missing으로 그대로 사용. KAN 대/중/소분류는 pandas Categorical dtype으로 유지해 LightGBM native categorical 처리.',
  dl: '연속형 feature(static/time-varying)는 train 기준 z-score로 표준화. KAN 대/중/소분류 3종은 train 기준 vocabulary로 정수 인덱스화(미확인 값은 <UNK>).',
};

function getFeatureInfo(model, horizon) {
  const h = HOLIDAY_FEATURES_BY_HORIZON[horizon] ? horizon : 'h1';
  const holidayCols = HOLIDAY_FEATURES_BY_HORIZON[h];
  const target = `${TARGET_COL_BY_HORIZON[h]} — week_st + ${h.slice(1)}주 시점의 raw 수요량`;

  if (DL_MODELS.has(model)) {
    const groups = [
      { label: 'Static · categorical', items: DL_STATIC_CATEGORICAL },
      { label: 'Static · continuous', items: DL_STATIC_CONTINUOUS },
      { label: 'Time-varying · known', items: [...DL_TIME_VARYING_KNOWN, ...holidayCols] },
      { label: 'Time-varying · observed', items: DL_TIME_VARYING_OBSERVED },
    ];
    return { groups, count: groups.reduce((s, g) => s + g.items.length, 0), target, preprocessing: PREPROCESSING_TEXT.dl };
  }

  const groups = [
    { label: 'Base', items: ML_BASE_FEATURES },
    { label: 'Demand summary', items: ML_DEMAND_SUMMARY_FEATURES },
    { label: 'Holiday (horizon별)', items: holidayCols },
  ];
  return {
    groups, count: groups.reduce((s, g) => s + g.items.length, 0), target,
    preprocessing: LGBM_FAMILY_MODELS.has(model) ? PREPROCESSING_TEXT.lgbm : PREPROCESSING_TEXT.rf,
  };
}

const TOOLTIP_BASE = { appendTo: 'body', confine: true, extraCssText: 'z-index:99999;' };

function fmtMetric(v) {
  if (v === null || v === undefined) return '-';
  return v.toFixed(2);
}

function niceBiasAxis(dataMin, biasBand) {
  const min = Math.floor(Math.min(dataMin, -biasBand) / 10) * 10;
  const max = biasBand + 5;
  return { min, max };
}

function niceLinearBounds(values, step) {
  const nums = values.filter((v) => v !== null && v !== undefined);
  if (!nums.length) return { min: 0, max: step };
  const min = Math.floor(Math.min(...nums) / step) * step;
  const max = Math.ceil(Math.max(...nums) / step) * step;
  return min === max ? { min: min - step, max: max + step } : { min, max };
}

function groupMarkAreaX() {
  return {
    silent: true,
    label: { color: '#94a3b8', fontSize: 11, fontWeight: 700, position: 'insideTop' },
    data: [
      [
        { xAxis: -0.5, itemStyle: { color: 'transparent' }, label: { formatter: '기존 모델' } },
        { xAxis: HURDLE_GROUP_BOUNDARY },
      ],
      [
        { xAxis: HURDLE_GROUP_BOUNDARY, itemStyle: { color: 'rgba(37,99,235,0.05)' }, label: { formatter: 'Hurdle' } },
        { xAxis: 6.5 },
      ],
    ],
  };
}

function groupMarkAreaXSubtle() {
  return {
    silent: true,
    data: [
      [{ xAxis: -0.5, itemStyle: { color: 'transparent' } }, { xAxis: HURDLE_GROUP_BOUNDARY }],
      [{ xAxis: HURDLE_GROUP_BOUNDARY, itemStyle: { color: 'rgba(37,99,235,0.04)' } }, { xAxis: 6.5 }],
    ],
  };
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

function filterByHorizon(text, horizon) {
  if (!text || !horizon || !text.includes(';')) return text;
  const parts = text.split(';').map((s) => s.trim());
  const match = parts.find((p) => p.toLowerCase().startsWith(`${horizon.toLowerCase()}:`));
  return match || text;
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

export default function MlDlAnalysis() {
  const [horizon, setHorizon] = useState('h1');
  const [trialFilter, setTrialFilter] = useState('ALL');
  const [selectedModel, setSelectedModel] = useState('Hurdle-LightGBM');
  const [selectedTrial, setSelectedTrial] = useState(null);

  const [trials, setTrials] = useState(null);
  const [trialsError, setTrialsError] = useState(null);
  const [boxplot, setBoxplot] = useState(null);
  const [boxplotError, setBoxplotError] = useState(null);
  const [passRate, setPassRate] = useState(null);
  const [passRateError, setPassRateError] = useState(null);
  const [improvement, setImprovement] = useState(null);
  const [improvementError, setImprovementError] = useState(null);
  const [hurdleCompare, setHurdleCompare] = useState(null);
  const [hurdleCompareError, setHurdleCompareError] = useState(null);
  const [modelDetail, setModelDetail] = useState(null);
  const [modelDetailError, setModelDetailError] = useState(null);

  useEffect(() => {
    let ignore = false;
    setTrialsError(null);
    api.getMlDlTrials({ horizon, model: 'ALL', trial_filter: trialFilter })
      .then((d) => { if (!ignore) setTrials(d); })
      .catch((e) => { if (!ignore) { setTrialsError(e.message); setTrials(null); } });
    return () => { ignore = true; };
  }, [horizon, trialFilter]);

  useEffect(() => {
    let ignore = false;
    setBoxplotError(null);
    api.getMlDlBiasBoxplot({ horizon })
      .then((d) => { if (!ignore) setBoxplot(d); })
      .catch((e) => { if (!ignore) { setBoxplotError(e.message); setBoxplot(null); } });
    return () => { ignore = true; };
  }, [horizon]);

  useEffect(() => {
    let ignore = false;
    setPassRateError(null);
    api.getMlDlPassRate({ horizon })
      .then((d) => { if (!ignore) setPassRate(d); })
      .catch((e) => { if (!ignore) { setPassRateError(e.message); setPassRate(null); } });
    return () => { ignore = true; };
  }, [horizon]);

  useEffect(() => {
    api.getMlDlImprovement().then(setImprovement).catch((e) => setImprovementError(e.message));
  }, []);

  useEffect(() => {
    api.getMlDlHurdleCompare().then(setHurdleCompare).catch((e) => setHurdleCompareError(e.message));
  }, []);

  useEffect(() => {
    let ignore = false;
    setModelDetailError(null);
    setModelDetail(null);
    api.getMlDlModelDetail({ model: MLDL_MODEL_LABELS[selectedModel], horizon })
      .then((d) => { if (!ignore) setModelDetail(d); })
      .catch((e) => { if (!ignore) { setModelDetailError(e.message); setModelDetail(null); } });
    return () => { ignore = true; };
  }, [selectedModel, horizon]);

  function handleHorizonChange(value) {
    setHorizon(value);
    setSelectedTrial(null);
  }

  function handleSelectModel(model) {
    setSelectedModel(model);
    setSelectedTrial(null);
  }

  function handleSelectTrial(point) {
    setSelectedModel(point.model);
    setSelectedTrial(point);
  }

  return (
    <div className="md-page">
      <div className="az-page-hd">
        <div>
          <h2>03 ML/DL 분석</h2>
          <p>과소예측 문제 → Hurdle 적용 → Bias 기준 통과 → H-RF/H-LGBM 비교 → H-LGBM 선정</p>
        </div>
        <div className="az-filter-bar">
          <div className="az-filter-group">
            <span className="az-filter-label">예측시점</span>
            <select className="az-filter-select" value={horizon} onChange={(e) => handleHorizonChange(e.target.value)}>
              {HORIZONS.map((h) => <option key={h.value} value={h.value}>{h.label}</option>)}
            </select>
          </div>
          <div className="az-filter-group">
            <span className="az-filter-label">모델</span>
            <select
              className="az-filter-select"
              value={MLDL_MODEL_LABELS[selectedModel]}
              onChange={(e) => handleSelectModel(LABEL_TO_MODEL[e.target.value])}
            >
              {MODELS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="md-charts">
        <div className="az-grid md-row-top">
          <TrialScatterCard
            data={trials} error={trialsError} selectedModel={selectedModel} selectedTrial={selectedTrial}
            onSelectTrial={handleSelectTrial} trialFilter={trialFilter} onTrialFilterChange={setTrialFilter}
          />
          <BiasBoxplotCard data={boxplot} error={boxplotError} selectedModel={selectedModel} onSelectModel={handleSelectModel} />
          <PassRateCard data={passRate} error={passRateError} selectedModel={selectedModel} onSelectModel={handleSelectModel} />
        </div>

        <div className="az-grid md-grid-bottom md-row-bottom">
          <ImprovementCard data={improvement} error={improvementError} />
          <HurdleCompareCard data={hurdleCompare} error={hurdleCompareError} />
          <DetailCard detail={modelDetail} error={modelDetailError} selectedTrial={selectedTrial} horizon={horizon} />
        </div>
      </div>
    </div>
  );
}

function TrialScatterCard({ data, error, selectedModel, selectedTrial, onSelectTrial, trialFilter, onTrialFilterChange }) {
  return (
    <div className="az-card">
      <div className="az-card-hd md-scatter-hd">
        <div>
          <h3>Trial 정확도·편향 분포</h3>
          <p>WAPE × Bias · 선택 모델을 진하게 강조</p>
        </div>
        <div className="md-seg">
          {TRIAL_FILTERS.map((t) => (
            <button
              key={t.value}
              className={`md-seg-btn ${trialFilter === t.value ? 'active' : ''}`}
              onClick={() => onTrialFilterChange(t.value)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <EChart
          fill
          option={buildScatterOption(data, selectedModel, selectedTrial)}
          onEvents={{
            click: (p) => {
              if (p.componentType === 'series' && p.data?.model !== undefined) {
                onSelectTrial({
                  model: p.data.model, trial: p.data.trial, wape: p.value[1], bias: p.value[0],
                  bias_pass: p.data.bias_pass, selected: p.data.selected,
                });
              }
            },
          }}
        />
      )}
    </div>
  );
}

function buildScatterOption(data, selectedModel, selectedTrial) {
  const byModel = {};
  data.points.forEach((p) => { (byModel[p.model] ??= []).push(p); });
  const dataMin = data.points.length ? Math.min(...data.points.map((p) => p.bias)) : -data.bias_band;
  const { min: xMin, max: xMax } = niceBiasAxis(dataMin, data.bias_band);

  const series = Object.entries(byModel).map(([model, pts]) => ({
    name: MLDL_MODEL_LABELS[model] || model,
    type: 'scatter',
    color: MODEL_COLORS[model],
    data: pts.map((p) => {
      const isSelTrial = selectedTrial && selectedTrial.model === model && selectedTrial.trial === p.trial;
      const isSelModel = model === selectedModel;
      return {
        value: [p.bias, p.wape],
        model: p.model, trial: p.trial, bias_pass: p.bias_pass, selected: p.selected, hp: p.hp,
        symbolSize: isSelTrial ? 18 : (isSelModel ? 10 : 7),
        itemStyle: {
          opacity: isSelTrial ? 1 : (isSelModel ? 0.95 : 0.16),
          borderColor: isSelTrial ? '#1e293b' : (p.selected ? '#1e293b' : 'transparent'),
          borderWidth: isSelTrial ? 2 : (p.selected ? 1.5 : 0),
        },
      };
    }),
  }));

  if (series.length > 0) {
    series[0].markArea = {
      silent: true,
      itemStyle: { color: 'rgba(191,219,254,0.25)' },
      label: { show: true, position: 'insideTop', color: '#2563eb', fontSize: 11, formatter: 'Bias 허용구간 |Bias|≤20%' },
      data: [[{ name: 'Bias 허용구간', xAxis: -data.bias_band }, { xAxis: data.bias_band }]],
    };
    series[0].markLine = {
      silent: true, symbol: 'none',
      lineStyle: { color: '#3b82f6', type: 'dashed', width: 1.3 },
      label: { formatter: (p) => `${p.value > 0 ? '+' : ''}${p.value}%`, color: '#3b82f6', fontSize: 11, position: 'insideEndTop' },
      data: [{ xAxis: data.bias_band }, { xAxis: -data.bias_band }],
    };
  }

  return {
    grid: { left: 56, right: 14, top: 24, bottom: 30 },
    legend: { top: 0, textStyle: { fontSize: 11, color: '#64748b' } },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => {
        const d = p.data;
        return `모델 ${MLDL_MODEL_LABELS[d.model] || d.model} · Trial ${d.trial}`
          + `<br/>예측시점 ${horizonLabel(data.horizon)}`
          + `<br/>WAPE ${fmtMetric(p.value[1])}% · Bias ${fmtMetric(p.value[0])}%`
          + `<br/>${d.bias_pass ? 'Bias 기준 통과' : 'Bias 기준 미통과'}`;
      },
    },
    xAxis: { type: 'value', name: 'Bias (%)', min: xMin, max: xMax, axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    yAxis: {
      type: 'value', name: 'WAPE (%)', nameLocation: 'middle', nameGap: 34, nameRotate: 90,
      axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } },
    },
    series,
  };
}

function BiasBoxplotCard({ data, error, selectedModel, onSelectModel }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>모델별 Bias 분포</h3>
        <p>
          0에 가까울수록 편향이 작음 · -20% 미만은 과소예측 기준 미충족
          <span className="md-group-caption">
            <span className="md-group-dot" />기존 모델
            <span className="md-group-dot md-group-dot-hurdle" />Hurdle
          </span>
        </p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <EChart
          fill
          option={buildBoxplotOption(data, selectedModel)}
          onEvents={{ click: (p) => { if (p.data?.model) onSelectModel(p.data.model); } }}
        />
      )}
    </div>
  );
}

function buildBoxplotOption(data, selectedModel) {
  const dataMin = data.entries.length ? Math.min(...data.entries.map((e) => e.box[0])) : -data.bias_band;
  const { min: yMin, max: yMax } = niceBiasAxis(dataMin, data.bias_band);

  return {
    grid: { left: 42, right: 14, top: 26, bottom: 24 },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => {
        const [min, q1, med, q3, max] = p.data.value;
        return `${p.name}<br/>Max ${fmtMetric(max)} · Q3 ${fmtMetric(q3)} · Med ${fmtMetric(med)}<br/>Q1 ${fmtMetric(q1)} · Min ${fmtMetric(min)}`;
      },
    },
    xAxis: {
      type: 'category', data: data.entries.map((e) => e.label),
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11.5, fontWeight: 600 },
    },
    yAxis: { type: 'value', name: 'Bias (%)', min: yMin, max: yMax, axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{
      type: 'boxplot',
      boxWidth: [20, 40],
      data: data.entries.map((e) => ({
        value: e.box,
        model: e.model,
        itemStyle: {
          color: e.model === selectedModel ? MODEL_COLORS[e.model] : '#e2e8f0',
          borderColor: e.model === selectedModel ? '#1e293b' : '#94a3b8',
          borderWidth: e.model === selectedModel ? 2 : 1,
        },
      })),
      markArea: groupMarkAreaXSubtle(),
      markLine: {
        silent: true, symbol: 'none',
        lineStyle: { color: '#64748b', type: 'dashed' },
        label: { formatter: (p) => `${p.value > 0 ? '+' : ''}${p.value}%`, color: '#64748b', fontSize: 11, position: 'insideEndTop' },
        data: [{ yAxis: data.bias_band }, { yAxis: -data.bias_band }],
      },
    }],
  };
}

function PassRateCard({ data, error, selectedModel, onSelectModel }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>Bias 기준 통과율</h3>
        <p>기존 모델에서는 통과 후보가 없었으나 Hurdle 적용 후 통과 후보가 생성됨</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <EChart
          fill
          option={buildPassRateOption(data, selectedModel)}
          onEvents={{ click: (p) => { if (p.data?.model) onSelectModel(p.data.model); } }}
        />
      )}
    </div>
  );
}

function buildPassRateOption(data, selectedModel) {
  return {
    grid: { left: 40, right: 12, top: 18, bottom: 26 },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => `${p.name}<br/>${Number.isFinite(p.data.pass_count) && Number.isFinite(p.data.total_count) ? `통과 수 ${p.data.pass_count} / 전체 수 ${p.data.total_count}<br/>` : ''}비율 ${fmtMetric(p.data.value)}%`,
    },
    xAxis: {
      type: 'category', data: data.entries.map((e) => e.label),
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11, interval: 0, rotate: 20 },
    },
    yAxis: { type: 'value', name: '비율(%)', min: 0, max: 100, axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [{
      type: 'bar',
      barMaxWidth: 34,
      markArea: groupMarkAreaX(),
      data: data.entries.map((e) => {
        const isHurdle = HURDLE_MODELS.has(e.model);
        return {
          value: e.pass_rate, pass_count: e.pass_count, total_count: e.total_count, model: e.model,
          itemStyle: {
            color: isHurdle ? MODEL_COLORS[e.model] : '#cbd5e1',
            opacity: e.model === selectedModel ? 1 : (isHurdle ? 0.85 : 0.55),
          },
          label: { show: true, position: 'top', formatter: () => `${Math.round(e.pass_rate)}%`, fontSize: 11, color: isHurdle ? '#475569' : '#94a3b8' },
        };
      }),
    }],
  };
}

function ImprovementCard({ data, error }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>LightGBM → Hurdle-LightGBM</h3>
        <p>기존 모델 → 판매 발생 여부를 분리한 2단계 모델</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && <ImprovementBody data={data} />}
    </div>
  );
}

function summarizeImprovement(data) {
  const biasGains = [];
  const wapeLosses = [];
  data.horizons.forEach((h) => {
    const o = data.old.points.find((p) => p.horizon === h);
    const n = data.new.points.find((p) => p.horizon === h);
    if (!o || !n) return;
    biasGains.push(Math.abs(o.bias) - Math.abs(n.bias));
    wapeLosses.push(n.wape - o.wape);
  });
  if (!biasGains.length) return null;
  const fmtRange = (arr, digits) => {
    const lo = Math.min(...arr).toFixed(digits);
    const hi = Math.max(...arr).toFixed(digits);
    return lo === hi ? lo : `${lo}~${hi}`;
  };
  return { biasText: fmtRange(biasGains, 2), wapeText: fmtRange(wapeLosses, 1) };
}

function ImprovementBody({ data }) {
  const horizons = data.horizons;
  const pick = (series, key) => horizons.map((h) => series.points.find((p) => p.horizon === h)?.[key] ?? null);
  const oldBias = pick(data.old, 'bias');
  const newBias = pick(data.new, 'bias');
  const oldWape = pick(data.old, 'wape');
  const newWape = pick(data.new, 'wape');
  const summary = summarizeImprovement(data);
  const biasBounds = niceLinearBounds([...oldBias, ...newBias], 5);
  const wapeBounds = niceLinearBounds([...oldWape, ...newWape], 1);

  return (
    <div className="md-mini-wrap">
      {summary && (
        <div className="md-improve-summary">
          |Bias| 감소 <strong>{summary.biasText}%p</strong> · WAPE 변화 <strong>+{summary.wapeText}%p</strong>
        </div>
      )}
      <div className="md-mini-block">
        <div className="md-mini-chart-title">Bias 변화 (0에 가까울수록 좋음)</div>
        <div className="md-mini-legend">
          <span className="md-legend-dot" style={{ '--dot-color': '#0d9488' }}>{data.old.label}</span>
          <span className="md-legend-dot" style={{ '--dot-color': '#2563eb' }}>{data.new.label}</span>
        </div>
        <EChart fill option={buildDumbbellOption(horizons, oldBias, newBias, '#0d9488', '#2563eb', biasBounds, 5)} />
      </div>
      <div className="md-mini-block">
        <div className="md-mini-chart-title">WAPE 변화 (%, 낮을수록 좋음)</div>
        <div className="md-mini-legend">
          <span className="md-legend-dot" style={{ '--dot-color': '#0d9488' }}>{data.old.label}</span>
          <span className="md-legend-dot" style={{ '--dot-color': '#2563eb' }}>{data.new.label}</span>
        </div>
        <EChart fill option={buildDumbbellOption(horizons, oldWape, newWape, '#0d9488', '#2563eb', wapeBounds, 1)} />
      </div>
    </div>
  );
}

function buildDumbbellOption(categories, oldVals, newVals, oldColor, newColor, bounds, step) {
  return {
    grid: { left: 52, right: 44, top: 4, bottom: 4 },
    xAxis: {
      type: 'value', min: bounds.min, max: bounds.max, interval: step,
      axisLabel: { color: '#94a3b8', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } },
    },
    yAxis: { type: 'category', data: categories.map(horizonLabel), inverse: true, axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => `${horizonLabel(categories[p.dataIndex])}<br/>${fmtMetric(oldVals[p.dataIndex])} → ${fmtMetric(newVals[p.dataIndex])}`,
    },
    series: [{
      type: 'custom',
      clip: false,
      renderItem: (params, api) => {
        const i = params.dataIndex;
        if (oldVals[i] === null || newVals[i] === null) return { type: 'group', children: [] };
        const p1 = api.coord([oldVals[i], i]);
        const p2 = api.coord([newVals[i], i]);
        return {
          type: 'group',
          children: [
            { type: 'line', shape: { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1] }, style: { stroke: '#cbd5e1', lineWidth: 2 } },
            { type: 'circle', shape: { cx: p1[0], cy: p1[1], r: 6 }, style: { fill: oldColor } },
            { type: 'circle', shape: { cx: p2[0], cy: p2[1], r: 6 }, style: { fill: newColor } },
            { type: 'text', style: { text: fmtMetric(oldVals[i]), x: p1[0], y: p1[1] - 13, fill: oldColor, fontSize: 12, fontWeight: 600, align: 'center' } },
            { type: 'text', style: { text: fmtMetric(newVals[i]), x: p2[0], y: p2[1] - 13, fill: newColor, fontSize: 12, fontWeight: 600, align: 'center' } },
          ],
        };
      },
      data: categories.map((_, i) => i),
    }],
  };
}

function HurdleCompareCard({ data, error }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>Hurdle 후보 최종 비교</h3>
        <p>Bias 통과 후보 간 최종 성능 비교</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && <HurdleCompareBody data={data} />}
    </div>
  );
}

function HurdleCompareBody({ data }) {
  const hrf = data.series.find((s) => s.model === 'Hurdle-RF');
  const hlgbm = data.series.find((s) => s.model === 'Hurdle-LightGBM');
  if (!hrf || !hlgbm) return <div className="az-hint">데이터가 없습니다.</div>;

  const rows = data.horizons.map((h) => ({
    horizon: h,
    hrf: hrf.points.find((p) => p.horizon === h),
    hlgbm: hlgbm.points.find((p) => p.horizon === h),
  }));
  const comparable = rows.filter((r) => r.hrf && r.hlgbm);
  const allLower = comparable.length > 0 && comparable.every((r) => r.hlgbm.wape < r.hrf.wape);
  const allHigher = comparable.length > 0 && comparable.every((r) => r.hlgbm.wape > r.hrf.wape);
  const conclusion = allLower
    ? `① Bias 기준: H-RF/H-LGBM 모두 통과 → ② WAPE 비교: ${comparable.map((r) => horizonLabel(r.horizon)).join("·")} 모두 H-LGBM 우세 → H-LGBM 선정`
    : allHigher
      ? `둘 다 Bias 기준 통과 → WAPE 더 낮은 ${hrf.label} 선정`
      : 'Horizon별로 더 낮은 WAPE를 보이는 후보가 다릅니다.';

  return (
    <>
      <table className="md-table">
        <thead>
          <tr><th rowSpan={2}>예측시점</th><th colSpan={2}>{hrf.label}</th><th colSpan={2}>{hlgbm.label}</th></tr>
          <tr><th>WAPE(%)</th><th>Bias(%)</th><th className="md-col-highlight">WAPE(%)</th><th>Bias(%)</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.horizon}>
              <td>{horizonLabel(r.horizon)}</td>
              <td>{fmtMetric(r.hrf?.wape)}</td>
              <td>{fmtMetric(r.hrf?.bias)}</td>
              <td className="md-col-highlight">{fmtMetric(r.hlgbm?.wape)}</td>
              <td>{fmtMetric(r.hlgbm?.bias)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="md-conclusion">{conclusion}</div>
    </>
  );
}

function modelOverview(detail) {
  return detail.track === 'hurdle'
    ? '판매 발생 여부와 발생 시 판매량을 분리해 예측하는 2단계 모델'
    : DL_MODELS.has(detail.model) ? '과거 시계열의 패턴을 학습해 미래 판매량을 예측하는 모델'
      : '판매 이력과 입력 변수의 관계를 학습해 판매량을 예측하는 모델';
}
function modelStructure(detail) {
  return detail.track === 'hurdle'
    ? '① 판매 발생 여부 분류 → ② 발생 시 판매량 예측 → ③ 결합 예측'
    : DL_MODELS.has(detail.model) ? '과거 시계열 입력 → 패턴 학습 → 판매량 예측' : '입력 변수 → 여러 결정 트리 학습 → 판매량 예측';
}
function modelReason(detail) {
  if (!detail.trial) return '선정 결과 데이터 없음';
  if (!detail.trial.bias_pass) return '미선정: Bias 허용 기준을 충족한 후보가 없음';
  if (detail.summary.some((r) => r.parameter_type === 'final_model')) return '선정: Bias 기준 통과 후, 1·2·4주 후 모두 H-RF보다 WAPE가 낮음';
  if (detail.model === 'Hurdle-RF') return '미선정: Bias 기준은 통과했으나 H-LGBM보다 WAPE가 높음';
  return 'Bias 기준 통과 후보';
}
function DetailCard({ detail, error, selectedTrial, horizon }) {
  const [modalOpen, setModalOpen] = useState(false);
  const clickedTrial = selectedTrial?.model === detail?.model ? selectedTrial : null;
  const trial = clickedTrial || detail?.trial;
  return (
    <div className={`az-card md-detail-card${detail ? ' md-detail-clickable' : ''}`}
      onClick={() => detail && setModalOpen(true)} role={detail ? 'button' : undefined}
      tabIndex={detail ? 0 : undefined}
      onKeyDown={(e) => { if (detail && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); setModalOpen(true); } }}>
      <div className="az-card-hd"><h3>선택 모델 요약</h3></div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !detail && <div className="az-hint">불러오는 중...</div>}
      {!error && detail && <>
        <div className="md-detail-title-row"><span className="az-detail-title">{detail.label}</span>
          {detail.trial && <span className={detail.trial.bias_pass ? 'md-pass-badge' : 'md-fail-badge'}>{detail.trial.bias_pass ? 'Bias 기준 통과' : 'Bias 기준 미통과'}</span>}
        </div>
        <p className="md-summary-description">{modelOverview(detail)}</p>
        <dl className="md-summary-list">
          <div><dt>예측 대상</dt><dd>{horizonLabel(horizon)} 판매량</dd></div>
          <div><dt>모델 구조</dt><dd>{modelStructure(detail)}</dd></div>
          <div><dt>선정 기준</dt><dd>Bias 기준 통과 → WAPE 최소 → 1%p 이내 동률권은 최악 검증 구간 WAPE 비교</dd></div>
          <div><dt>선정/미선정 이유</dt><dd>{modelReason(detail)}</dd></div>
        </dl>
        {trial && <div className="md-trial-box md-summary-metrics">
          <div>{horizonLabel(horizon)} · {clickedTrial ? '선택' : '대표'} Trial {trial.trial}</div>
          <div>WAPE <strong>{fmtMetric(trial.wape)}%</strong> · Bias <strong>{fmtMetric(trial.bias)}%</strong></div>
          <div>판정 <span className={trial.bias_pass ? 'md-pass-badge' : 'md-fail-badge'}>{trial.bias_pass ? 'Bias 기준 통과' : 'Bias 기준 미통과'}</span></div>
        </div>}
      </>}
      {modalOpen && detail && <ModelDetailModal key={`${detail.model}-${horizon}`} detail={detail} horizon={horizon} onClose={() => setModalOpen(false)} />}
    </div>
  );
}

const FEATURE_HELP = {
  '입수': ['포장당 수량', '상품 단위 특성 반영'],
  'KAN_대분류': ['상품 대분류', '상품군 차이 반영'], 'KAN_중분류': ['상품 중분류', '상품군 차이 반영'], 'KAN_소분류': ['상품 소분류', '상품군 차이 반영'],
  'ISO_주차': ['연중 주차', '주 단위 계절성 반영'], '월': ['달', '월별 패턴 반영'], '분기': ['분기', '분기별 패턴 반영'],
  '평균온도': ['주간 평균 기온', '날씨와 수요 관계 반영'], '총강수량': ['주간 누적 강수량', '강수와 수요 관계 반영'],
  'qty_log1p': ['판매량의 로그 변환값', '판매 이력 반영'],
  'qty_lag1_filled_log1p': ['직전 판매량의 결측 처리·로그 변환값', '최근 수요 반영'],
  'qty_rollmean_4_filled_log1p': ['4주 이동평균의 결측 처리·로그 변환값', '최근 수요 수준 반영'],
  'qty_rollstd_4_filled_log1p': ['4주 이동표준편차의 결측 처리·로그 변환값', '수요 변동성 반영'],
  'existed_before_regime': ['기준 체계 이전 존재 여부', '상품 이력 특성 반영'],
  'is_warmup': ['초기 이력 구간 여부', '이력 축적 상태 반영'], 'coldstart_flag': ['신규 이력 여부', '이력이 짧은 상품 구분'],
  'adi_expanding_filled': ['누적 평균 수요 발생 간격', '수요 간헐성 반영'], 'cv2_expanding_filled': ['누적 수요 변동계수 제곱', '수요 변동성 반영'],
  '강수량_호우_count': ['호우 발생 횟수', '집중 강수 영향 반영'], 'covid_flag': ['코로나 시기 여부', '시기별 수요 차이 반영'],
  'center_is_B': ['B센터 여부', '센터 차이 반영'], 'temp_x_precip': ['기온과 강수량 교호항', '복합 날씨 영향 반영'],
  'center_temp_inter': ['센터와 기온 교호항', '센터별 기온 영향 반영'], 'weeks_since_last_active_filled': ['마지막 판매 이후 경과 주수', '판매 공백 반영'],
  'ccsi_lag_m1': ['전월 소비자심리지수', '소비 심리 반영'], 'cpi_y1_prev': ['이전 소비자물가지수', '물가 수준 반영'], 'cpi_y2_prev_yoy': ['이전 물가의 전년 대비 변화', '물가 변화 반영'],
};
function featureHelp(name) {
  return FEATURE_HELP[name] || (name.includes('공휴일_W') ? ['예측 주 또는 인접 주의 공휴일 변수', '공휴일 주변 수요 패턴 반영'] : ['설명 데이터 없음', '확인된 설명 없음']);
}
function parameterHelp(name) {
  const key = name.toLowerCase();
  const stage = key.startsWith('cls_') || key.startsWith('classifier.') ? '판매 발생 분류: ' : key.startsWith('reg_') || key.startsWith('regressor.') ? '판매량 예측: ' : '';
  const meanings = { max_features: '트리 분기에 사용하는 변수 비율', min_samples_leaf: '말단 노드의 최소 학습 표본 수', num_leaves: '트리의 최대 말단 노드 수', min_child_samples: '말단 노드의 최소 학습 표본 수', hidden_size: '신경망 내부 표현의 크기', n_heads: '동시에 학습하는 주의집중 패턴 수', target_transform: '예측 대상 판매량의 변환 방식' };
  return stage + (Object.entries(meanings).find(([key]) => name.toLowerCase().includes(key))?.[1] || '설명 데이터 없음');
}

const ALL_HORIZON_KEYS = ['h1', 'h2', 'h4'];

function ModelDetailModal({ detail, horizon, onClose }) {
  const [horizonDetails, setHorizonDetails] = useState({ [horizon]: detail });

  useEffect(() => {
    let ignore = false;
    const others = ALL_HORIZON_KEYS.filter((h) => h !== horizon);
    Promise.all(
      others.map((h) =>
        api.getMlDlModelDetail({ model: MLDL_MODEL_LABELS[detail.model], horizon: h })
          .then((d) => [h, d])
          .catch(() => [h, null]),
      ),
    ).then((pairs) => {
      if (ignore) return;
      setHorizonDetails((prev) => {
        const next = { ...prev, [horizon]: detail };
        pairs.forEach(([h, d]) => { next[h] = d; });
        return next;
      });
    });
    return () => { ignore = true; };
  }, [detail, horizon]);

  useEffect(() => {
    function onKeyDown(e) { if (e.key === 'Escape') onClose(); }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const isHurdle = detail.track === 'hurdle';
  const baseLearner = detail.model.replace('Hurdle-', '');
  const hpoRow = detail.summary.find((r) => r.parameter_type === 'hpo');
  const fixedRow = detail.summary.find((r) => r.parameter_type === 'fixed');
  const finalRow = detail.summary.find((r) => r.parameter_type === 'final_model');
  const featureInfo = getFeatureInfo(detail.model, horizon);

  return createPortal(
    <div className="md-modal-backdrop" onClick={(e) => { e.stopPropagation(); onClose(); }} onKeyDown={(e) => e.stopPropagation()}>
      <div className="md-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="md-modal-hd">
          <div className="md-modal-hd-title">
            <h2>{detail.label}</h2>
            {detail.trial && (
              <span className={detail.trial.bias_pass ? 'md-pass-badge' : 'md-fail-badge'}>
                {detail.trial.bias_pass ? 'Bias 기준 통과' : 'Bias 기준 미통과'}
              </span>
            )}
          </div>
          <button type="button" className="md-modal-close" onClick={onClose} aria-label="닫기">✕</button>
        </div>

        <div className="md-modal-body">
          <section className="md-modal-section"><div className="md-modal-section-hd">모델 개요</div><p className="md-detail-caption">{modelOverview(detail)}</p></section>
          <div className="md-modal-section">
            <div className="md-modal-section-hd">모델 구조</div>
            {isHurdle ? (
              <div className="md-hurdle-diagram md-hurdle-diagram-lg">
                <div className="md-hurdle-box md-hurdle-box-cls">
                  <div className="md-hurdle-box-title">① 판매 발생 여부 분류</div>
                  <div className="md-hurdle-box-sub">{baseLearner}</div>
                </div>
                <span className="md-hurdle-plus">+</span>
                <div className="md-hurdle-box md-hurdle-box-reg">
                  <div className="md-hurdle-box-title">② 발생 시 판매량 예측</div>
                  <div className="md-hurdle-box-sub">{baseLearner}</div>
                </div>
                <span className="md-hurdle-arrow">→</span>
                <div className="md-hurdle-box md-hurdle-box-out">
                  <div className="md-hurdle-box-title">결합 예측</div>
                  <div className="md-hurdle-box-sub">최종 수요</div>
                </div>
              </div>
            ) : (
              <div className="md-detail-caption">
                {DL_MODELS.has(detail.model) ? `Lookback 시퀀스 인코더 (${detail.label})` : `${detail.label} 단일 회귀 모델`} — 고정 구조는 아래 &lsquo;고정값&rsquo; 참고
              </div>
            )}
          </div>

          <div className="md-modal-section">
            <div className="md-modal-section-hd">예측 대상</div>
            <div className="md-detail-caption">{featureInfo.target}</div>
          </div>

          <div className="md-modal-section">
            <div className="md-modal-section-hd">입력 Feature ({featureInfo.count}개)</div>
            <table className="md-table md-reference-table">
              <thead><tr><th>변수명</th><th>의미</th><th>활용 목적</th></tr></thead>
              <tbody>{featureInfo.groups.flatMap((g) => g.items).map((name) => <tr key={name}><td><code>{name}</code></td><td>{featureHelp(name)[0]}</td><td>{featureHelp(name)[1]}</td></tr>)}</tbody>
            </table>
          </div>

          <div className="md-modal-section">
            <div className="md-modal-section-hd">전처리</div>
            <div className="md-detail-caption">{featureInfo.preprocessing}</div>
          </div>

          {hpoRow && (
            <div className="md-modal-section">
              <div className="md-modal-section-hd">HPO 파라미터 · 모델 설정 탐색</div>
              <table className="md-table md-reference-table">
                <thead><tr><th>파라미터명</th><th>의미</th><th>선택값 (제공된 경우)</th></tr></thead>
                <tbody>{hpoRow.parameter.split(/[,/]/).map((name) => name.trim()).filter(Boolean).map((name) => {
                  const entry = Object.entries(detail.trial?.hp || {}).find(([key]) => key.toLowerCase() === name.toLowerCase());
                  return <tr key={name}><td><code>{name}</code></td><td>{parameterHelp(name)}</td><td>{entry ? String(entry[1]) : '아래 API 원문 참고'}</td></tr>;
                })}</tbody>
              </table>
              {hpoRow.candidates_or_rule && <div className="md-detail-row-line"><span className="md-detail-row-key">탐색 범위</span><span>{hpoRow.candidates_or_rule}</span></div>}
              {hpoRow.selected_or_fixed && <div className="md-detail-row-line"><span className="md-detail-row-key">선택 결과</span><span>{hpoRow.selected_or_fixed}</span></div>}
              <div className="md-detail-caption">선정 순서: Bias 기준 통과 → WAPE 최소 → 1%p 이내 동률권은 최악 검증 구간 WAPE 비교</div>
            </div>
          )}

          {fixedRow && (
            <div className="md-modal-section">
              <div className="md-modal-section-hd">고정값</div>
              <KVGrid text={fixedRow.selected_or_fixed} />
            </div>
          )}

          <section className="md-modal-section"><div className="md-modal-section-hd">선정/탈락 이유</div><p className="md-detail-caption">{modelReason(detail)}</p></section>
          {finalRow && (
            <div className="md-modal-section">
              <div className="md-modal-section-hd">최종 선정 근거</div>
              <div className="md-detail-caption">{finalRow.selected_or_fixed}</div>
            </div>
          )}

          <div className="md-modal-section">
            <div className="md-modal-section-hd">예측시점별 결과</div>
            <table className="md-table">
              <thead>
                <tr><th>예측시점</th><th>Trial</th><th>WAPE(%)</th><th>Bias(%)</th><th>판정</th></tr>
              </thead>
              <tbody>
                {ALL_HORIZON_KEYS.map((h) => {
                  const t = horizonDetails[h]?.trial;
                  return (
                    <tr key={h}>
                      <td>{horizonLabel(h)}</td>
                      <td>{t ? t.trial : '불러오는 중...'}</td>
                      <td>{t ? t.wape.toFixed(2) : '-'}</td>
                      <td>{t ? t.bias.toFixed(2) : '-'}</td>
                      <td>
                        {t && (
                          <span className={t.bias_pass ? 'md-pass-badge' : 'md-fail-badge'}>
                            {t.bias_pass ? 'Bias 기준 통과' : 'Bias 기준 미통과'}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
