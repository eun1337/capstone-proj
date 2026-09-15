import { useEffect, useState } from 'react';
import { api } from '../../api/client.js';
import EChart from '../../components/charts/EChart.jsx';
import './analysis.css';
import './ModelComparison.css';

const CENTERS = [
  { value: 'ALL', label: 'ALL' },
  { value: 'A', label: 'A' },
  { value: 'B', label: 'B' },
];
const HORIZONS = [
  { value: 'ALL', label: '전체' },
  { value: 'h1', label: 'h1' },
  { value: 'h2', label: 'h2' },
  { value: 'h4', label: 'h4' },
];
const SCOPES = [
  { value: 'full_common', label: 'Full Common' },
  { value: 'model_fit', label: 'Model-fit' },
  { value: 'fallback', label: 'Fallback' },
];
const METRICS = [
  { value: 'WAPE', label: 'WAPE' },
  { value: 'Bias', label: 'Bias' },
  { value: 'MAE', label: 'MAE' },
  { value: 'RMSE', label: 'RMSE' },
];

const STAT_COLOR = '#3b82f6';
const ML_COLOR = '#8b5cf6';
const TOOLTIP_BASE = { appendTo: 'body', confine: true, extraCssText: 'z-index:99999;' };

function fmtMetric(v, metric) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  return metric === 'MAE' || metric === 'RMSE' ? v.toFixed(2) : `${v.toFixed(1)}%`;
}

function fmtDiff(v, metric) {
  const unit = metric === 'WAPE' || metric === 'Bias' ? '%p' : '';
  const sign = v >= 0 ? '' : '-';
  return `${sign}${Math.abs(v).toFixed(1)}${unit}`;
}

function fmtPct(v) {
  return `${Math.round(v)}%`;
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

export default function ModelComparison() {
  const [center, setCenter] = useState('ALL');
  const [horizon, setHorizon] = useState('ALL');
  const [metric, setMetric] = useState('WAPE');
  const [scope, setScope] = useState('full_common');
  const [selectedDemandGroup, setSelectedDemandGroup] = useState('Q4');

  const [bar, setBar] = useState(null);
  const [barError, setBarError] = useState(null);
  const [kpi, setKpi] = useState(null);
  const [kpiError, setKpiError] = useState(null);
  const [heatmap, setHeatmap] = useState(null);
  const [heatmapError, setHeatmapError] = useState(null);
  const [winnerShare, setWinnerShare] = useState(null);
  const [winnerShareError, setWinnerShareError] = useState(null);
  const [wapeProfile, setWapeProfile] = useState(null);
  const [wapeProfileError, setWapeProfileError] = useState(null);
  const [demandShare, setDemandShare] = useState(null);
  const [demandShareError, setDemandShareError] = useState(null);
  const [demandDetail, setDemandDetail] = useState(null);
  const [demandDetailError, setDemandDetailError] = useState(null);

  useEffect(() => {
    let ignore = false;
    setBarError(null);
    api.getFinalBar({ center, horizon, metric, scope })
      .then((d) => { if (!ignore) setBar(d); })
      .catch((e) => { if (!ignore) { setBarError(e.message); setBar(null); } });
    return () => { ignore = true; };
  }, [center, horizon, metric, scope]);

  useEffect(() => {
    let ignore = false;
    setKpiError(null);
    api.getFinalKpi({ center, horizon, metric, scope })
      .then((d) => { if (!ignore) setKpi(d); })
      .catch((e) => { if (!ignore) { setKpiError(e.message); setKpi(null); } });
    return () => { ignore = true; };
  }, [center, horizon, metric, scope]);

  useEffect(() => {
    let ignore = false;
    setHeatmapError(null);
    api.getFinalHeatmap({ metric, scope })
      .then((d) => { if (!ignore) setHeatmap(d); })
      .catch((e) => { if (!ignore) { setHeatmapError(e.message); setHeatmap(null); } });
    return () => { ignore = true; };
  }, [metric, scope]);

  useEffect(() => {
    let ignore = false;
    setWinnerShareError(null);
    api.getFinalWinnerShare({ center, horizon })
      .then((d) => { if (!ignore) setWinnerShare(d); })
      .catch((e) => { if (!ignore) { setWinnerShareError(e.message); setWinnerShare(null); } });
    return () => { ignore = true; };
  }, [center, horizon]);

  useEffect(() => {
    let ignore = false;
    setWapeProfileError(null);
    api.getFinalWapeProfile({ center, horizon })
      .then((d) => { if (!ignore) setWapeProfile(d); })
      .catch((e) => { if (!ignore) { setWapeProfileError(e.message); setWapeProfile(null); } });
    return () => { ignore = true; };
  }, [center, horizon]);

  useEffect(() => {
    let ignore = false;
    setDemandShareError(null);
    api.getFinalDemandShare({ center, horizon })
      .then((d) => { if (!ignore) setDemandShare(d); })
      .catch((e) => { if (!ignore) { setDemandShareError(e.message); setDemandShare(null); } });
    return () => { ignore = true; };
  }, [center, horizon]);

  useEffect(() => {
    let ignore = false;
    setDemandDetailError(null);
    api.getFinalDemandDetail({ center, horizon, quartile: selectedDemandGroup })
      .then((d) => { if (!ignore) setDemandDetail(d); })
      .catch((e) => { if (!ignore) { setDemandDetailError(e.message); setDemandDetail(null); } });
    return () => { ignore = true; };
  }, [center, horizon, selectedDemandGroup]);

  return (
    <div className="mc-page">
      <div className="az-page-hd">
        <div>
          <h2>04 최종 모델 비교</h2>
          <p>2024 Holdout에서 SARIMA와 Hurdle-LightGBM 중 어떤 모델이 더 우수했는가</p>
        </div>
        <div className="az-filter-bar">
          <div className="az-filter-group">
            <span className="az-filter-label">Center</span>
            <select className="az-filter-select" value={center} onChange={(e) => setCenter(e.target.value)}>
              {CENTERS.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
            </select>
          </div>
          <div className="az-filter-group">
            <span className="az-filter-label">Horizon</span>
            <select className="az-filter-select" value={horizon} onChange={(e) => setHorizon(e.target.value)}>
              {HORIZONS.map((h) => <option key={h.value} value={h.value}>{h.label}</option>)}
            </select>
          </div>
          <div className="az-filter-group">
            <span className="az-filter-label">분석 범위</span>
            <select className="az-filter-select" value={scope} onChange={(e) => setScope(e.target.value)}>
              {SCOPES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
            </select>
          </div>
          <div className="az-filter-group">
            <span className="az-filter-label">평가 지표</span>
            <select className="az-filter-select" value={metric} onChange={(e) => setMetric(e.target.value)}>
              {METRICS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        </div>
      </div>

      <div className="mc-charts">
        <div className="az-grid mc-row-top">
          <BarCard data={bar} error={barError} metric={metric} />
          <KpiCard data={kpi} error={kpiError} metric={metric} />
          <HeatmapCard data={heatmap} error={heatmapError} metric={metric} center={center} horizon={horizon} />
        </div>

        <div className="az-grid mc-row-mid">
          <WinnerShareCard data={winnerShare} error={winnerShareError} selected={selectedDemandGroup} onSelect={setSelectedDemandGroup} />
          <WapeProfileCard data={wapeProfile} error={wapeProfileError} selected={selectedDemandGroup} />
        </div>

        <div className="az-grid mc-row-bottom">
          <DemandShareCard data={demandShare} error={demandShareError} />
          <DemandDetailCard data={demandDetail} error={demandDetailError} />
        </div>
      </div>
    </div>
  );
}

function BarCard({ data, error, metric }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>SARIMA vs H-LGBM (2024)</h3>
        <p>동일 조건(Center·Scope) h1/h2/h4 {metric} 비교</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && data.points.length === 0 && <div className="az-hint">데이터가 없습니다.</div>}
      {!error && data && data.points.length > 0 && <EChart fill option={buildBarOption(data, metric)} />}
    </div>
  );
}

function buildBarOption(data, metric) {
  const cats = data.points.map((p) => p.horizon);
  return {
    grid: { left: 42, right: 12, top: 30, bottom: 26 },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: '#475569' } },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: {
      type: 'category', data: cats,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 12 },
    },
    yAxis: {
      type: 'value', name: metric === 'MAE' || metric === 'RMSE' ? metric : `${metric}(%)`,
      nameTextStyle: { fontSize: 10, color: '#94a3b8' },
      axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } },
    },
    series: [
      {
        name: 'SARIMA', type: 'bar', barMaxWidth: 40, itemStyle: { color: STAT_COLOR, borderRadius: [3, 3, 0, 0] },
        label: { show: true, position: 'top', formatter: (p) => fmtMetric(p.value, metric), fontSize: 11, color: '#475569' },
        data: data.points.map((p) => p.stat_value),
      },
      {
        name: 'H-LGBM', type: 'bar', barMaxWidth: 40, itemStyle: { color: ML_COLOR, borderRadius: [3, 3, 0, 0] },
        label: { show: true, position: 'top', formatter: (p) => fmtMetric(p.value, metric), fontSize: 11, color: '#475569' },
        data: data.points.map((p) => p.ml_value),
      },
    ],
  };
}

function kpiDiff(e, metric) {
  return metric === 'Bias' ? Math.abs(e.stat_value) - Math.abs(e.ml_value) : e.diff;
}

function KpiCard({ data, error, metric }) {
  return (
    <div className="az-card mc-kpi-card">
      <div className="az-card-hd">
        <h3>성능 차이</h3>
        <p>SARIMA 대비 H-LGBM {metric === 'Bias' ? '절대 Bias 개선폭' : '개선폭'}</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && data.entries.length === 0 && <div className="az-hint">데이터가 없습니다.</div>}
      {!error && data && data.entries.length > 0 && (
        <div className="mc-kpi-list">
          {data.entries.map((e) => {
            const diff = kpiDiff(e, metric);
            const improving = diff >= 0;
            const worsening = diff < 0;
            return (
              <div className="mc-kpi-row" key={e.horizon}>
                <span className="mc-kpi-h">{e.horizon}</span>
                <span className={`mc-kpi-diff ${improving ? 'is-good' : worsening ? 'is-bad' : 'is-neutral'}`}>
                  {improving && '▼ '}{worsening && '▲ '}{fmtDiff(diff, metric)}
                </span>
                <span className="mc-kpi-detail">({fmtMetric(e.stat_value, metric)} → {fmtMetric(e.ml_value, metric)})</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function heatmapDiff(c, metric) {
  return metric === 'Bias' ? Math.abs(c.stat_value) - Math.abs(c.ml_value) : c.diff;
}

function HeatmapCard({ data, error, metric, center, horizon }) {
  const hasNegative = !!data && data.cells.some((c) => heatmapDiff(c, metric) < 0);
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <h3>Center × Horizon Heatmap</h3>
        <p>
          {metric === 'Bias'
            ? '센터별·예측기간별 절대 Bias 개선폭 (|SARIMA-Bias| − |H-LGBM-Bias|)'
            : `센터별·예측기간별 성능 차이 (SARIMA-${metric} − H-LGBM-${metric})`}
        </p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <>
          <EChart fill option={buildHeatmapOption(data, metric, center, horizon, hasNegative)} />
          {hasNegative ? (
            <div className="mc-scale-bar">
              <span>SARIMA 우세</span>
              <span className="mc-scale-gradient" />
              <span>H-LGBM 우세</span>
            </div>
          ) : (
            <div className="mc-scale-bar">
              <span>차이 없음</span>
              <span className="mc-scale-gradient mc-scale-gradient-seq" />
              <span>H-LGBM 개선폭 큼</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function buildHeatmapOption(data, metric, selCenter, selHorizon, hasNegative) {
  const diffs = data.cells.map((c) => heatmapDiff(c, metric));
  const bound = robustBound(diffs);
  const domainMin = hasNegative ? -bound : 0;
  const cells = data.cells.map((c) => {
    const d = heatmapDiff(c, metric);
    return {
      value: [data.horizons.indexOf(c.horizon), data.centers.indexOf(c.center), Math.max(domainMin, Math.min(bound, d))],
      raw: c, diff: d, center: c.center, horizon: c.horizon,
      itemStyle: (c.center === selCenter || c.horizon === selHorizon) && !(selCenter === 'ALL' && selHorizon === 'ALL')
        ? { borderColor: '#1e293b', borderWidth: 2 } : undefined,
    };
  });

  return {
    grid: { left: 34, right: 16, top: 10, bottom: 26 },
    tooltip: {
      ...TOOLTIP_BASE,
      formatter: (p) => `${p.data.center} · ${p.data.horizon}<br/>SARIMA: ${fmtMetric(p.data.raw.stat_value, metric)}<br/>H-LGBM: ${fmtMetric(p.data.raw.ml_value, metric)}<br/>${metric === 'Bias' ? '절대 Bias 개선폭' : '차이'}: ${fmtDiff(p.data.diff, metric)}`,
    },
    xAxis: {
      type: 'category', data: data.horizons,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11 },
    },
    yAxis: {
      type: 'category', data: data.centers, inverse: true,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 11 },
    },
    visualMap: {
      min: domainMin, max: bound, show: false,
      inRange: { color: hasNegative ? ['#fecaca', '#f8fafc', '#93c5fd', '#1d4ed8'] : ['#eff6ff', '#1d4ed8'] },
    },
    series: [{
      type: 'heatmap',
      data: cells,
      label: { show: true, formatter: (p) => fmtDiff(p.data.diff, metric), fontSize: 11, color: '#1e293b' },
      emphasis: { itemStyle: { borderColor: '#1e293b', borderWidth: 2 } },
    }],
  };
}

function WinnerShareCard({ data, error, selected, onSelect }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <div className="mc-card-hd-row">
          <h3>수요규모별 Winner Share</h3>
          <span className="mc-badge-fc">Full Common 기준</span>
        </div>
        <p>Q1~Q4(수요량 4등분) 구간별 우세 모델 비율</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <EChart
          fill
          option={buildWinnerShareOption(data, selected)}
          onEvents={{ click: (p) => { if (p.name) onSelect(p.name); } }}
        />
      )}
    </div>
  );
}

function winnerShareBarData(values, entries, selected, color) {
  return values.map((v, i) => {
    const isSel = entries[i].quartile === selected;
    return {
      value: v,
      itemStyle: isSel
        ? { color, opacity: 1, borderColor: '#1e293b', borderWidth: 2 }
        : { color, opacity: 0.6 },
    };
  });
}

function buildWinnerShareOption(data, selected) {
  const cats = data.entries.map((e) => e.quartile);
  return {
    grid: { left: 36, right: 12, top: 30, bottom: 20 },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: '#475569' } },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: {
      type: 'category', data: cats,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 12, fontWeight: 700 },
    },
    yAxis: { type: 'value', max: 100, name: '비율(%)', nameTextStyle: { fontSize: 10, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [
      {
        name: 'SARIMA', type: 'bar', stack: 'w', barMaxWidth: 46,
        label: { show: true, formatter: (p) => (p.value > 8 ? `${Math.round(p.value)}%` : ''), color: '#fff', fontSize: 11, fontWeight: 700 },
        data: winnerShareBarData(data.entries.map((e) => e.sarima_share), data.entries, selected, STAT_COLOR),
      },
      {
        name: 'H-LGBM', type: 'bar', stack: 'w', barMaxWidth: 46,
        label: { show: true, formatter: (p) => (p.value > 8 ? `${Math.round(p.value)}%` : ''), color: '#fff', fontSize: 11, fontWeight: 700 },
        data: winnerShareBarData(data.entries.map((e) => e.hlgbm_share), data.entries, selected, ML_COLOR),
      },
    ],
  };
}

function WapeProfileCard({ data, error, selected }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <div className="mc-card-hd-row">
          <h3>수요규모별 WAPE Profile</h3>
          <span className="mc-badge-fc">Full Common 기준</span>
        </div>
        <p>Q1~Q4 구간별 실제 수요량 가중 WAPE 변화 · 굵은 점 = 선택 구간</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && <EChart fill option={buildWapeProfileOption(data, selected)} />}
    </div>
  );
}

function wapeProfileData(values, selIdx) {
  return values.map((v, i) => (i === selIdx ? { value: v, symbolSize: 14 } : { value: v, symbolSize: 7 }));
}

function buildWapeProfileOption(data, selected) {
  const cats = data.entries.map((e) => e.quartile);
  const selIdx = cats.indexOf(selected);
  return {
    grid: { left: 40, right: 16, top: 20, bottom: 20 },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: '#475569' } },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis' },
    xAxis: {
      type: 'category', data: cats, boundaryGap: true,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#64748b', fontSize: 12, fontWeight: 700 },
    },
    yAxis: { type: 'value', name: 'WAPE(%)', nameTextStyle: { fontSize: 10, color: '#94a3b8' }, axisLabel: { color: '#64748b', fontSize: 11 }, splitLine: { lineStyle: { color: '#f1f5f9' } } },
    series: [
      {
        name: 'SARIMA', type: 'line', lineStyle: { width: 2, color: STAT_COLOR }, itemStyle: { color: STAT_COLOR },
        label: { show: true, position: 'top', formatter: (p) => p.value.toFixed(1), fontSize: 11, color: STAT_COLOR },
        markArea: selIdx >= 0 ? {
          silent: true,
          itemStyle: { color: 'rgba(37,99,235,0.06)' },
          data: [[{ xAxis: selIdx - 0.5 }, { xAxis: selIdx + 0.5 }]],
        } : undefined,
        data: wapeProfileData(data.entries.map((e) => e.stat_wape), selIdx),
      },
      {
        name: 'H-LGBM', type: 'line', lineStyle: { width: 2, color: ML_COLOR }, itemStyle: { color: ML_COLOR },
        label: { show: true, position: 'bottom', formatter: (p) => p.value.toFixed(1), fontSize: 11, color: ML_COLOR },
        data: wapeProfileData(data.entries.map((e) => e.ml_wape), selIdx),
      },
    ],
  };
}

function DemandShareCard({ data, error }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <div className="mc-card-hd-row">
          <h3>SKU 수 vs 실제 수요량 비중</h3>
          <span className="mc-badge-fc">Full Common 기준</span>
        </div>
        <p>단순 SKU 개수 → 실제 수요량 반영 시 우세 비중 변화 (WAPE 기준)</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && <EChart fill option={buildDemandShareOption(data)} />}
    </div>
  );
}

function buildDemandShareOption(data) {
  const cats = ['SKU 개수 기준', '실제 수요량 기준'];
  return {
    grid: { left: 88, right: 30, top: 26, bottom: 10 },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 11, color: '#475569' } },
    tooltip: { ...TOOLTIP_BASE, trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'value', max: 100, show: false },
    yAxis: {
      type: 'category', data: cats, inverse: true,
      axisLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#475569', fontSize: 12, fontWeight: 700 },
    },
    series: [
      {
        name: 'SARIMA', type: 'bar', stack: 'd', barMaxWidth: 34, itemStyle: { color: STAT_COLOR },
        label: { show: true, formatter: (p) => `${Math.round(p.value)}%`, color: '#fff', fontSize: 12, fontWeight: 700 },
        data: [data.sku_basis.sarima_pct, data.demand_basis.sarima_pct],
      },
      {
        name: 'H-LGBM', type: 'bar', stack: 'd', barMaxWidth: 34, itemStyle: { color: ML_COLOR },
        label: { show: true, formatter: (p) => `${Math.round(p.value)}%`, color: '#fff', fontSize: 12, fontWeight: 700 },
        data: [data.sku_basis.hlgbm_pct, data.demand_basis.hlgbm_pct],
      },
    ],
  };
}

function DemandDetailCard({ data, error }) {
  return (
    <div className="az-card">
      <div className="az-card-hd">
        <div className="mc-card-hd-row">
          <h3>선택 구간 상세</h3>
          <span className="mc-badge-fc">Full Common 기준</span>
        </div>
        <p>Winner Share에서 선택한 수요구간의 해석</p>
      </div>
      {error && <div className="az-hint az-hint-error">{error}</div>}
      {!error && !data && <div className="az-hint">불러오는 중...</div>}
      {!error && data && (
        <div className="mc-detail-body">
          <div className="mc-detail-badge">{data.quartile}</div>
          <div className="az-detail-row">
            <span className="az-detail-label">수요 특성</span>
            <span className="az-detail-value">{data.label} (SKU {data.n_skus.toLocaleString()}건)</span>
          </div>
          <div className="az-detail-row">
            <span className="az-detail-label">Winner</span>
            <span className="az-detail-value">
              SKU 기준 SARIMA {fmtPct(data.sku_share.sarima_pct)} · H-LGBM {fmtPct(data.sku_share.hlgbm_pct)}
            </span>
          </div>
          <div className="az-detail-row">
            <span className="az-detail-label">수요량 비중</span>
            <span className="az-detail-value">
              SARIMA {fmtPct(data.demand_share.sarima_pct)} · H-LGBM {fmtPct(data.demand_share.hlgbm_pct)}
            </span>
          </div>
          <div className="az-detail-row">
            <span className="az-detail-label">해석</span>
            <span className="az-detail-value">{data.interpretation}</span>
          </div>
        </div>
      )}
    </div>
  );
}
