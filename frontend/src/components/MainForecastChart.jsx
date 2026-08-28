import { useMemo } from 'react';
import './MainForecastChart.css';

// ── Config ─────────────────────────────────────────────────────
const W     = 640;
const H     = 150;
const PAD_L = 54;
const PAD_R = 18;
const PAD_T = 10;
const PAD_B = 24;
const EW    = W - PAD_L - PAD_R;
const EH    = H - PAD_T - PAD_B;
const DAY_MS = 86400000;

const HORIZON_DEFS = [
  { key: 'h1', label: '1주 후' },
  { key: 'h2', label: '2주 후' },
  { key: 'h4', label: '4주 후' },
];

function sxDate(dateStr, minDate, maxDate) {
  const span = (new Date(maxDate) - new Date(minDate)) / DAY_MS || 1;
  const off  = (new Date(dateStr) - new Date(minDate)) / DAY_MS;
  return PAD_L + (off / span) * EW;
}

const sy = (v, min, max) => {
  const rng = max - min || 1;
  return PAD_T + (1 - (v - min) / rng) * EH;
};

function shortDate(dateStr) {
  const d = new Date(dateStr);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

// 메인 "수요예측 추이" 차트 — SKU 미선택 시 GET /demand-trend(집계, 같은 단위 SKU 합산),
// SKU 선택 시 GET /forecast(해당 SKU) 응답을 그대로 쓴다. 두 경우 모두 h1/h2/h4는 기존
// 산출물의 값 그대로이며, h3를 만들거나 h2~h4 사이를 보간하지 않는다 — 실제 날짜 간격
// 그대로 두 점을 직선으로 이을 뿐이다(2주~4주 사이 구간이 1주~2주보다 넓게 보이는 이유).
export default function MainForecastChart({ trend, loading, error, unit }) {
  const chart = useMemo(() => {
    if (!trend) return null;
    const history = trend.history ?? [];

    const horizons = HORIZON_DEFS
      .map((h) => {
        const v = trend[h.key];
        return v ? { ...h, ...v } : null;
      })
      .filter(Boolean);

    const histPoints = history.map((h) => ({
      date: h.week_st,
      v: h.sales_qty,
      dateLabel: shortDate(h.week_st),
      isPred: false,
      title: `${h.week_st} · 실제 ${Math.round(h.sales_qty).toLocaleString()}${unit}`,
    }));

    const predPoints = horizons.map((h) => ({
      date: h.target_date,
      v: h.predicted_qty,
      dateLabel: h.label,
      isPred: true,
      title: `${h.target_date} · 예측 ${h.predicted_qty.toFixed(1)}${unit}`,
    }));

    const allPoints = [...histPoints, ...predPoints];
    if (allPoints.length === 0) return { histPoints, predPoints, allPoints, minDate: null, maxDate: null, minV: 0, maxV: 1 };

    const minDate = allPoints[0].date;
    const maxDate = allPoints[allPoints.length - 1].date;

    const vals = allPoints.map((p) => p.v);
    const minV = Math.min(...vals, 0);
    const maxV = (Math.max(...vals) || 0) * 1.15 || 1;

    return { histPoints, predPoints, allPoints, minDate, maxDate, minV, maxV };
  }, [trend, unit]);

  if (loading) {
    return <div className="mfc-empty"><p>수요예측 추이를 불러오는 중...</p></div>;
  }
  if (error) {
    return (
      <div className="mfc-empty mfc-empty-error">
        <p>표시할 데이터가 없습니다.</p>
        <p className="mfc-empty-sub">{error}</p>
      </div>
    );
  }
  if (!chart || chart.histPoints.length === 0) {
    return <div className="mfc-empty"><p>표시할 실적 데이터가 없습니다.</p></div>;
  }

  const { histPoints, predPoints, allPoints, minDate, maxDate, minV, maxV } = chart;
  const bottom = PAD_T + EH;
  const lastHist = histPoints[histPoints.length - 1];
  const divX = sxDate(lastHist.date, minDate, maxDate);

  const px = (p) => sxDate(p.date, minDate, maxDate);
  const xy = (p) => `${px(p).toFixed(1)},${sy(p.v, minV, maxV).toFixed(1)}`;

  const histLine = histPoints.map(xy).join(' ');
  const histArea = [...histPoints.map(xy), `${divX.toFixed(1)},${bottom}`, `${PAD_L},${bottom}`].join(' ');

  const predLinePts = predPoints.length ? [lastHist, ...predPoints].map(xy).join(' ') : '';
  const predArea = predPoints.length
    ? [`${divX.toFixed(1)},${bottom}`, ...predPoints.map(xy), `${px(predPoints[predPoints.length - 1]).toFixed(1)},${bottom}`].join(' ')
    : '';

  const yTickCount = 4;
  const yStep = (maxV - minV) / yTickCount || 1;
  const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => Math.round(minV + i * yStep));

  return (
    <div className="mfc-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', overflow: 'visible' }}>
        <defs>
          <linearGradient id="mfc-hg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#2563eb" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#2563eb" stopOpacity="0" />
          </linearGradient>
        </defs>

        {yTicks.map((v) => {
          const y = sy(v, minV, maxV);
          return (
            <g key={v}>
              <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} stroke="#f1f5f9" strokeWidth="1" />
              <text x={PAD_L - 8} y={y + 4} textAnchor="end" fontSize="10.5" fill="#94a3b8">{v.toLocaleString()}</text>
            </g>
          );
        })}

        {allPoints.map((p, idx) => (
          <text
            key={idx}
            x={px(p)}
            y={H - 8}
            textAnchor="middle"
            fontSize={p.isPred ? 10 : 9.5}
            fontWeight={p.isPred ? 700 : 400}
            fill={p.isPred ? '#2563eb' : '#94a3b8'}
          >
            {p.dateLabel}
          </text>
        ))}

        {predPoints.length > 0 && <polygon points={predArea} fill="#eff6ff" opacity="0.75" />}

        <polygon points={histArea} fill="url(#mfc-hg)" />
        <polyline points={histLine} fill="none" stroke="#2563eb" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />

        {predPoints.length > 0 && (
          <polyline
            points={predLinePts}
            fill="none"
            stroke="#60a5fa"
            strokeWidth="2"
            strokeDasharray="7,4"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}

        {histPoints.map((p, idx) => (
          <circle key={`h-${idx}`} cx={px(p)} cy={sy(p.v, minV, maxV)} r="2.8" fill="#2563eb">
            <title>{p.title}</title>
          </circle>
        ))}
        {predPoints.map((p, idx) => (
          <circle key={`p-${idx}`} cx={px(p)} cy={sy(p.v, minV, maxV)} r="4" fill="#fff" stroke="#60a5fa" strokeWidth="2">
            <title>{p.title}</title>
          </circle>
        ))}

        <line x1={divX} y1={PAD_T} x2={divX} y2={bottom} stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="4,3" />
        <rect x={divX - 24} y={PAD_T - 2} width="48" height="15" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="0.8" />
        <text x={divX} y={PAD_T + 9} textAnchor="middle" fontSize="9.5" fill="#475569" fontWeight="700">예측 시작</text>

        <line x1={PAD_L} y1={bottom} x2={W - PAD_R} y2={bottom} stroke="#e2e8f0" />
        <line x1={PAD_L} y1={PAD_T}  x2={PAD_L}     y2={bottom}  stroke="#e2e8f0" />
      </svg>

      <div className="mfc-legend">
        <span className="mfc-leg-item">
          <span className="mfc-leg-line solid" />
          실제 판매수량
        </span>
        <span className="mfc-leg-item">
          <span className="mfc-leg-line dashed" />
          AI 예상수요
        </span>
      </div>
    </div>
  );
}
