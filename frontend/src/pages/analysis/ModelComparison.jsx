import { useState } from 'react';
import EChart from '../../components/charts/EChart.jsx';
import dashboard from '../../../../data/dashboard/final_model_comparison/final_model_summary.json';
import './analysis.css';
import './ModelComparison.css';

const COLORS = ['#3b82f6', '#9333ea', '#cbd5e1'];
const METRICS = ['WAPE', 'Bias', 'MAE', 'RMSE', 'MASE'];
const CENTERS = [['ALL', '전체'], ['A', 'A센터'], ['B', 'B센터']];
const HORIZONS = ['1', '2', '4'];
const Q_NAMES = ['저수요', '중저수요', '중고수요', '고수요'];
const SCOPE_NAMES = { model_fit: '정상 SARIMA 적용', fallback: '대체 예측 적용', constant: '상수 예측 적용', full_coverage: '전체 운영' };
const SCOPE_HELP = {
  model_fit: '저장된 forecast_source=sarima인 실제 모델 적용 row. Development 대체 여부로 제외하지 않음.',
  fallback: 'forecast_source가 sarima/constant가 아닌 naive_mean 및 실제 대체 예측 override row.',
  constant: 'Development 기준 상수예측: 이력이 없으면 0, 수요가 일정하면 해당 상수값. forecast-origin 기준 신규 SKU와 다름.',
  full_coverage: '정상 적용 + 대체 예측 + 상수 예측의 동일 예측 row 교집합 전체.',
};
const TIP = { appendTo: 'body', confine: true, backgroundColor: '#fff', borderColor: '#cbd5e1', textStyle: { color: '#334155', fontSize: 12 }, extraCssText: 'z-index:10000;box-shadow:0 5px 20px #0f172a18;' };
const fmt = (n) => !Number.isFinite(n) ? '—' : Math.abs(n) >= 1e7 ? n.toExponential(2) : n.toLocaleString('ko-KR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const unit = (metric) => ['WAPE', 'Bias'].includes(metric) ? '%' : '';
const guide = (metric) => metric === 'Bias' ? '0에 가까울수록 좋음' : metric === 'MASE' ? '낮을수록 좋음 · 1: Development naive 오차 수준' : '낮을수록 좋음';
const value = (row, model, metric) => row?.[`${model}_${metric}`];
const improvement = (row, metric) => {
  const a = value(row, 'stat', metric), b = value(row, 'ml', metric);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  const displayedA = Number(a.toFixed(2));
  const displayedB = Number(b.toFixed(2));
  return metric === 'Bias' ? Math.abs(displayedA) - Math.abs(displayedB) : displayedA - displayedB;
};
const metricText = (row, metric) => `SARIMA ${fmt(value(row, 'stat', metric))}${unit(metric)} · H-LGBM ${fmt(value(row, 'ml', metric))}${unit(metric)}`;
const CONSTANT_DETAILS = [['constant_no_development_rows', 'Development 이력 없음'], ['constant_with_development_rows', 'Development 이력 있음'], ['constant_zero_rows', 'constant=0'], ['constant_positive_rows', 'constant>0']];
function contextTip(row, metric) {
  if (!row) return '데이터 없음';
  const q = row.quartile ? `<br/>${row.quartile}: ${Q_NAMES[Number(row.quartile[1]) - 1]} · 수요 범위 ${fmt(row.range_low)}~${fmt(row.range_high)}` : '';
  const mase = metric === 'MASE' ? `<br/>MASE 유효 SKU ${fmt(row.mase_valid_skus)} / 전체 ${fmt(row.n_skus)}<br/>제외 ${fmt(row.mase_excluded_skus)} (${fmt(row.mase_excluded_pct)}%) · 유효 row ${fmt(row.mase_valid_rows)}` : '';
  const formula = metric === 'WAPE' ? 'Σ|예측−실제| ÷ Σ실제 ×100' : metric === 'Bias' ? '(Σ예측−Σ실제) ÷ Σ실제 ×100' : metric === 'MAE' ? 'Σ|예측−실제| ÷ row 수' : metric === 'RMSE' ? '√(Σ오차² ÷ row 수)' : 'Σ(|오차| ÷ Development naive scale) ÷ 유효 row 수';
  return `${row.scope ? `${SCOPE_HELP[row.scope]}<br/>교집합 키: center·SKU·forecast origin·target date·horizon<br/>계산: ${formula}<br/>` : ''}${metricText(row, metric)}<br/>비교 row ${fmt(row.n_rows)} · SKU ${fmt(row.n_skus)}<br/>실제 수요량 ${fmt(row.actual_sum)}${q}${mase}${row.scope === 'constant' ? CONSTANT_DETAILS.map(([key, label]) => `<br/>${label}: ${fmt(row[key])} row`).join('') : ''}`;
}
function records(name, center, horizon) {
  return dashboard[name].filter((r) => r.center === center && (horizon === undefined || r.horizon === horizon));
}
function useFilters() {
  const [center, setCenter] = useState('ALL');
  const [metric, setMetric] = useState('WAPE');
  const [horizon, setHorizon] = useState('1');
  return { center, setCenter, metric, setMetric, horizon, setHorizon };
}
function Filters({ filters: f, horizon = true }) {
  return <div className="fc-filters">
    <select className="fc-filter-menu" aria-label="센터 선택" value={f.center} onChange={(e) => f.setCenter(e.target.value)}>{CENTERS.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
    <select className="fc-filter-menu" aria-label="평가 지표 선택" value={f.metric} onChange={(e) => f.setMetric(e.target.value)}>{METRICS.map((m) => <option key={m} value={m}>{m}</option>)}</select>
    {horizon && <select className="fc-filter-menu" aria-label="예측시점 선택" value={f.horizon} onChange={(e) => f.setHorizon(e.target.value)}>{HORIZONS.map((h) => <option key={h} value={h}>{h}주 후</option>)}</select>}
  </div>;
}
function Card({ number, title, children, subtitle, actions, className = '', ...props }) {
  return <section className={`az-card fc-card ${className}`} {...props}><header><div className="fc-card-heading"><div><h3>{number}. {title}</h3>{subtitle && <p>{subtitle}</p>}</div>{actions}</div></header>{children}</section>;
}
function seriesOption(rows, metric, categories, type = 'bar') {
  if (type === 'bar') return {
    color: COLORS,
    grid: { left: 62, right: 42, top: 30, bottom: 24 },
    legend: { top: 0, data: ['SARIMA', 'H-LGBM'], textStyle: { fontSize: 10 } },
    tooltip: { ...TIP, trigger: 'axis', formatter: (ps) => `${categories[ps[0].dataIndex]}<br/>${contextTip(rows[ps[0].dataIndex], metric)}` },
    xAxis: { type: 'value', axisLabel: { fontSize: 10, formatter: fmt }, splitLine: { lineStyle: { color: '#eef2f6' } } },
    yAxis: { type: 'category', inverse: true, data: categories, axisLabel: { fontSize: 10 }, axisTick: { show: false } },
    series: ['stat', 'ml'].map((model, i) => ({ name: i ? 'H-LGBM' : 'SARIMA', type: 'bar', barMaxWidth: 18, data: rows.map((r) => value(r, model, metric) ?? null), label: { show: true, position: 'right', color: COLORS[i], fontSize: 10, formatter: (p) => fmt(p.value) }, ...(metric === 'MASE' && i === 0 ? { markLine: { silent: true, symbol: 'none', data: [{ xAxis: 1 }], label: { formatter: '1' }, lineStyle: { color: '#94a3b8' } } } : {}) })),
  };
  return {
    color: COLORS,
    grid: { left: 48, right: 14, top: 30, bottom: 40 },
    legend: { top: 0, data: ['SARIMA', 'H-LGBM'], textStyle: { fontSize: 10 } },
    tooltip: { ...TIP, trigger: 'axis', formatter: (ps) => `${categories[ps[0].dataIndex]}<br/>${contextTip(rows[ps[0].dataIndex], metric)}` },
    xAxis: { type: 'category', data: categories, axisLabel: { fontSize: 10, interval: 0 }, axisTick: { show: false } },
    yAxis: { type: 'value', axisLabel: { fontSize: 10, formatter: fmt }, splitLine: { lineStyle: { color: '#eef2f6' } } },
    series: ['stat', 'ml'].map((model, i) => ({ name: i ? 'H-LGBM' : 'SARIMA', type, barMaxWidth: 26, symbolSize: 7,
      data: rows.map((r) => value(r, model, metric) ?? null),
      label: { show: type === 'bar', position: 'top', color: COLORS[i], fontSize: 10, formatter: (p) => fmt(p.value) },
      ...(metric === 'MASE' && i === 0 ? { markLine: { silent: true, symbol: 'none', data: [{ yAxis: 1 }], label: { formatter: '1' }, lineStyle: { color: '#94a3b8' } } } : {}),
    })),
  };
}
function OverallCard({ filters: f }) {
  const rows = HORIZONS.map((h) => records('overall_metrics_2024', f.center, h)[0]);
  return <Card number={1} title="2024 최종 성능 비교" actions={<Filters filters={f} horizon={false} />}><EChart fill option={seriesOption(rows, f.metric, HORIZONS.map((h) => `${h}주 후`))} />
    <div className="fc-improvement-strip">{rows.map((r, i) => <div key={HORIZONS[i]}><span>{HORIZONS[i]}주 후</span><strong>{fmt(improvement(r, f.metric))}{['WAPE', 'Bias'].includes(f.metric) ? '%p' : ''}</strong>{f.metric === 'Bias' && <small>|Bias| 감소</small>}</div>)}</div></Card>;
}
function WinnerShareCard({ filters: f }) {
  const rows = records('demand_type_winner_share_2024', f.center, f.horizon).filter((r) => r.metric === f.metric);
  const demandLabels = rows.map((r) => Q_NAMES[Number(r.quartile[1]) - 1]);
  const displayRows = rows.map((r) => { const total = r.stat_sku_count + r.ml_sku_count; return { ...r, stat_display: total ? r.stat_sku_count / total * 100 : null, ml_display: total ? r.ml_sku_count / total * 100 : null }; });
  const option = {
    color: COLORS.slice(0, 2), grid: { left: 40, right: 12, top: 28, bottom: 30 },
    legend: { top: 0, textStyle: { fontSize: 10 } },
    tooltip: { ...TIP, trigger: 'axis', formatter: (ps) => { const r = displayRows[ps[0].dataIndex]; return `${r.quartile}<br/>${contextTip(r, f.metric)}<br/>우세 SKU 수: SARIMA ${fmt(r.stat_sku_count)} · H-LGBM ${fmt(r.ml_sku_count)}<br/>우세 판정 SKU 기준: SARIMA ${fmt(r.stat_display)}% · H-LGBM ${fmt(r.ml_display)}%`; } },
    xAxis: { type: 'category', data: demandLabels, axisLabel: { interval: 0, fontSize: 9.5, lineHeight: 13 } }, yAxis: { type: 'value', max: 100, axisLabel: { formatter: (v) => `${fmt(v)}%` } },
    series: [
      { name: 'H-LGBM', type: 'bar', stack: 'share', barMaxWidth: 46, data: displayRows.map((r) => r.ml_display), itemStyle: { color: COLORS[1] }, label: { show: true, position: 'inside', color: '#fff', fontSize: 10, fontWeight: 700, formatter: (p) => p.value >= 8 ? `${fmt(p.value)}%` : '' } },
      { name: 'SARIMA', type: 'bar', stack: 'share', barMaxWidth: 46, data: displayRows.map((r) => r.stat_display), itemStyle: { color: COLORS[0] }, label: { show: true, position: 'inside', color: '#fff', fontSize: 10, formatter: (p) => p.value >= 8 ? `${fmt(p.value)}%` : '' } },
    ],
  };
  return <Card number={2} title="수요구간별 우세 상품 비율" actions={<Filters filters={f} />}><EChart fill option={option} /></Card>;
}
function DemandCard({ filters: f }) {
  const rows = records('demand_type_metrics_2024', f.center, f.horizon);
  const labels = rows.map((r) => Q_NAMES[Number(r.quartile[1]) - 1]);
  return <Card number={3} title="수요구간별 실제 오차" subtitle={`동일 예측 row 합산 ${f.metric} · ${guide(f.metric)}`} actions={<Filters filters={f} />}><EChart fill option={seriesOption(rows, f.metric, labels, 'line')} /><div className="fc-demand-highlight" title="센터×상품×예측기간별 2024 실제수요 합계 → 수요가 적은 순으로 정렬 → 상품 수가 비슷하도록 4등분">고수요 구간 · 전체 실제수요의 약 89.7%</div></Card>;
}
function niceStep(value) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const ratio = value / power;
  return (ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 5 ? 5 : 10) * power;
}
function coverageOption(rows, metric) {
  const horizontal = metric === 'WAPE';
  const categories = rows.map((r) => SCOPE_NAMES[r.scope].replace(' ', '\n'));
  const values = rows.flatMap((r) => [value(r, 'stat', metric), value(r, 'ml', metric)]).filter(Number.isFinite);
  const magnitudes = values.map(Math.abs).sort((a, b) => a - b);
  const largest = magnitudes.at(-1) || 1;
  const second = magnitudes.at(-2) || largest;
  const hasExtreme = largest > Math.max(second * 2.2, second + 10);
  const referenceValues = hasExtreme ? values.filter((v) => Math.abs(v) < largest) : values;
  const isBias = metric === 'Bias';
  let min;
  let max;
  let interval;
  if (isBias) {
    const bound = Math.max(...referenceValues.map(Math.abs), 1) * 1.2;
    interval = niceStep((bound * 2) / 5);
    max = Math.ceil(bound / interval) * interval;
    min = -max;
  } else {
    const lo = Math.min(...referenceValues, 0);
    const hi = Math.max(...referenceValues, 1);
    const padding = Math.max((hi - lo) * .18, hi * .05, 1);
    interval = niceStep((hi - lo + padding * 2) / 5);
    min = Math.max(0, Math.floor((Math.min(...referenceValues) - padding) / interval) * interval);
    max = Math.ceil((hi + padding) / interval) * interval;
  }
  const clip = (actual) => {
    if (!Number.isFinite(actual)) return { value: null, actual, clipped: false };
    const clipped = actual > max || actual < min;
    return { value: Math.min(max - interval * .12, Math.max(min + interval * .12, actual)), actual, clipped };
  };
  return {
    color: COLORS.slice(0, 2),
    grid: horizontal ? { left: 88, right: 36, top: 30, bottom: 24 } : { left: 50, right: 14, top: 30, bottom: 42 },
    legend: { top: 0, data: ['SARIMA', 'H-LGBM'], textStyle: { fontSize: 10 } },
    tooltip: { ...TIP, trigger: 'axis', formatter: (ps) => `${categories[ps[0].dataIndex]}<br/>${contextTip(rows[ps[0].dataIndex], metric)}` },
    xAxis: horizontal ? { type: 'value', min, max, interval, axisLabel: { fontSize: 9.5, formatter: fmt }, splitLine: { lineStyle: { color: '#e9eef5' } } } : { type: 'category', data: categories, axisLabel: { fontSize: 9.5, interval: 0 }, axisTick: { show: false } },
    yAxis: horizontal ? { type: 'category', inverse: true, data: categories, axisLabel: { fontSize: 9.5 }, axisTick: { show: false } } : { type: 'value', min, max, interval, axisLabel: { fontSize: 9.5, formatter: fmt }, splitLine: { lineStyle: { color: '#e9eef5' } } },
    series: ['stat', 'ml'].map((model, i) => ({
      name: i ? 'H-LGBM' : 'SARIMA', type: 'bar', barMaxWidth: 25,
      data: rows.map((r) => clip(value(r, model, metric))),
      label: { show: true, position: horizontal ? 'right' : 'top', color: COLORS[i], fontSize: 9.5, fontWeight: 700, formatter: (p) => p.data.clipped ? `${fmt(p.data.actual)} ↑` : fmt(p.data.actual) },
      ...(isBias && i === 0 ? { markLine: { silent: true, symbol: 'none', label: { show: false }, data: [{ yAxis: 0 }], lineStyle: { color: '#94a3b8' } } } : {}),
    })),
  };
}
function CoverageCard({ filters: f }) {
  const [open, setOpen] = useState(false);
  const rows = records('coverage_scope_metrics_2024', f.center, f.horizon).filter((row) => row.scope !== 'full_coverage');
  return <Card number={4} title="실제 운영 조건별 성능" className="fc-clickable-card" role="button" tabIndex={0} aria-haspopup="dialog" actions={<Filters filters={f} />} onClick={(event) => { if (!event.target.closest('select')) setOpen(true); }} onKeyDown={(event) => { if (event.target === event.currentTarget && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); setOpen(true); } }}><EChart fill option={coverageOption(rows, f.metric)} />{open && <div className="fc-modal-backdrop" onClick={(event) => { event.stopPropagation(); setOpen(false); }}><section className="fc-modal fc-scope-modal" role="dialog" aria-modal="true" aria-labelledby="fc-scope-title" onClick={(event) => event.stopPropagation()}><header><div><h3 id="fc-scope-title">비교 구간 정의와 계산 방식</h3><p>2024 Holdout · 동일 예측 row 기준</p></div><button type="button" className="fc-modal-close" aria-label="닫기" onClick={() => setOpen(false)}>×</button></header><div className="fc-scope-modal-body">{Object.entries(SCOPE_NAMES).map(([key, label]) => <div key={key}><b>{label}</b><span>{SCOPE_HELP[key]}</span></div>)}<div className="is-method"><b>비교 방식</b><span>center·SKU·forecast origin·target date·horizon가 모두 같은 row만 교집합으로 묶고 선택 지표를 다시 계산합니다.</span></div><p>forecast-origin 기준 신규 SKU는 두 모델의 동일 평가 row가 없어 직접 비교하지 않습니다.</p></div></section></div>}</Card>;
}
function SuitabilityCard({ filters: f }) {
  const q = records('demand_type_winner_share_2024', f.center, f.horizon).filter((r) => r.metric === f.metric);
  const skuCounts = [q.reduce((s, r) => s + r.stat_sku_count, 0), q.reduce((s, r) => s + r.ml_sku_count, 0)];
  const demandSums = [q.reduce((s, r) => s + (r.stat_demand_sum || 0), 0), q.reduce((s, r) => s + (r.ml_demand_sum || 0), 0)];
  const skuTotal = skuCounts[0] + skuCounts[1];
  const demandTotal = demandSums[0] + demandSums[1];
  const shares = [skuCounts.map((v) => skuTotal ? v / skuTotal * 100 : 0), demandSums.map((v) => demandTotal ? v / demandTotal * 100 : 0)];
  const categories = ['상품 수 기준\n우세 비율', '우세 상품의\n실제 수요량 비중'];
  const option = {
    color: COLORS.slice(0, 2), grid: { left: 42, right: 18, top: 32, bottom: 32 },
    legend: { top: 0, data: ['SARIMA', 'H-LGBM'], textStyle: { fontSize: 10 } },
    tooltip: { ...TIP, trigger: 'axis', formatter: (ps) => { const i = ps[0].dataIndex; return `${categories[i]}<br/>SARIMA ${fmt(shares[i][0])}% · H-LGBM ${fmt(shares[i][1])}%<br/>${i === 0 ? `우세 SKU ${fmt(skuCounts[0])} / ${fmt(skuCounts[1])}` : `우세 상품의 실제 수요량 ${fmt(demandSums[0])} / ${fmt(demandSums[1])}`}<br/>${f.metric === 'Bias' ? '|Bias|' : f.metric}가 더 우수한 모델 기준`; } },
    xAxis: { type: 'category', data: categories, axisTick: { show: false }, axisLabel: { fontSize: 10 } },
    yAxis: { type: 'value', max: 100, axisLabel: { formatter: '{value}%' }, splitLine: { lineStyle: { color: '#eef2f6' } } },
    series: [
      { name: 'H-LGBM', type: 'bar', stack: 'total', barMaxWidth: 54, data: shares.map((row) => row[1]), itemStyle: { color: COLORS[1] }, label: { show: true, position: 'inside', color: '#fff', formatter: (p) => p.value >= 8 ? `${fmt(p.value)}%` : '', fontSize: 10 } },
      { name: 'SARIMA', type: 'bar', stack: 'total', barMaxWidth: 54, data: shares.map((row) => row[0]), itemStyle: { color: COLORS[0] }, label: { show: true, position: 'inside', color: '#fff', formatter: (p) => p.value >= 8 ? `${fmt(p.value)}%` : '', fontSize: 10 } },
    ],
  };
  return <Card number={5} title="상품 수와 실제 수요량 기준 비교" actions={<Filters filters={f} />}><EChart fill option={option} /></Card>;
}
function ConclusionCard({ filters: f }) {
  const holdout = HORIZONS.map((h) => records('overall_metrics_2024', f.center, h)[0]);
  const centerLabel = CENTERS.find(([key]) => key === f.center)?.[1] || f.center;
  const winsAll = holdout.every((row) => improvement(row, 'WAPE') > 0);
  const demandRows = records('demand_type_metrics_2024', f.center, f.horizon);
  const highDemand = demandRows.find((row) => row.quartile === 'Q4');
  const demandTotal = demandRows.reduce((sum, row) => sum + (row.actual_sum || 0), 0);
  const highDemandShare = highDemand && demandTotal ? highDemand.actual_sum / demandTotal * 100 : null;
  const highDemandGain = improvement(highDemand, 'WAPE');
  const highDemandWin = Number.isFinite(highDemandGain) && highDemandGain > 0;
  const fullCoverage = records('coverage_scope_metrics_2024', f.center, f.horizon).find((row) => row.scope === 'full_coverage');
  return <Card number={6} title="2024 최종 평가 요약"><div className="fc-holdout-evidence is-primary"><div className="fc-high-demand-reason"><div><span>① 전반적인 예측 성능</span><strong>A/B센터 · 1·2·4주 후 모두<br />H-LGBM의 WAPE가 더 낮음</strong><small>센터와 예측기간이 달라도 일관된 성능</small></div><div><span>② 물동량이 큰 상품 강점</span><strong>고수요 상품 WAPE 20.70%p 감소<br />89.83% → 69.13%</strong><small>고수요 구간이 전체 실제수요의 <b>89.7%</b> 차지</small></div></div></div></Card>;
}
export default function ModelComparison() {
  const filters = useFilters();
  return <div className="fc-page"><div className="az-page-hd"><div><h2>04 최종 모델 비교</h2><p>2024 Holdout · 통계 대표모델 SARIMA의 실제 예측과 ML/DL 최종모델 H-LGBM의 동일 row 검증</p></div></div><div className="fc-grid"><OverallCard filters={filters} /><WinnerShareCard filters={filters} /><DemandCard filters={filters} /><CoverageCard filters={filters} /><SuitabilityCard filters={filters} /><ConclusionCard filters={filters} /></div></div>;
}
