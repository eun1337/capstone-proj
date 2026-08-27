import { useMemo } from 'react';
import './ForecastChart.css';

// ── Config ─────────────────────────────────────────────────────
const W     = 540;
const H     = 168;
const PAD_L = 48;
const PAD_R = 18;
const PAD_T = 14;
const PAD_B = 30;
const EW    = W - PAD_L - PAD_R;
const EH    = H - PAD_T - PAD_B;
const HIST  = 60; // fixed historical days

// ── Deterministic data gen ─────────────────────────────────────
function genData(futureDays) {
  const NOISE = [14,-9,18,-6,22,-13,8,-19,26,-11,16,-7,20,-10,24,-15,6,-21,17,-8,12,-16,21,-5,25,-12,10,-18,15,-4,19,-14,23,-10,7,-17,27,-6,11,-13];
  const pts = [];
  let v = 520;
  for (let i = 0; i < HIST + futureDays; i++) {
    const n = NOISE[i % NOISE.length] + Math.sin(i / 8) * 18;
    v = Math.max(180, Math.min(870, v + n + (i < HIST ? 1.5 : 0.5)));
    pts.push({ i, v: Math.round(v), isPred: i >= HIST });
  }
  return pts;
}

// ── Scales ─────────────────────────────────────────────────────
const sx = (i, total) => PAD_L + (i / (total - 1)) * EW;
const sy = (v, min, max) => PAD_T + (1 - (v - min) / (max - min)) * EH;

// ── Date label helper ──────────────────────────────────────────
function dateLabel(dayOffset) {
  const d = new Date();
  d.setDate(d.getDate() + dayOffset);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

// ── Component ──────────────────────────────────────────────────
export default function ForecastChart({ forecastDays = 30 }) {
  const data = useMemo(() => genData(forecastDays), [forecastDays]);
  const total = data.length;

  const vals = data.map((d) => d.v);
  const minV = Math.min(...vals) * 0.91;
  const maxV = Math.max(...vals) * 1.06;

  const hist = data.filter((d) => !d.isPred);
  const pred = data.slice(HIST - 1); // include last hist point for continuity

  const histPts = hist.map((d) => `${sx(d.i, total).toFixed(1)},${sy(d.v, minV, maxV).toFixed(1)}`);
  const predPts = pred.map((d) => `${sx(d.i, total).toFixed(1)},${sy(d.v, minV, maxV).toFixed(1)}`);

  const divX   = sx(HIST - 1, total);
  const bottom = PAD_T + EH;

  // Area polygon strings
  const histArea = [
    ...histPts,
    `${sx(HIST - 1, total).toFixed(1)},${bottom}`,
    `${PAD_L},${bottom}`,
  ].join(' ');

  const predArea = [
    ...predPts,
    `${sx(total - 1, total).toFixed(1)},${bottom}`,
    `${divX.toFixed(1)},${bottom}`,
  ].join(' ');

  // Y ticks
  const yTickCount = 4;
  const yStep = (maxV - minV) / yTickCount;
  const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) =>
    Math.round(minV + i * yStep)
  );

  // X ticks
  const xTickIdxs = [0, 15, 30, 45, HIST - 1, HIST + Math.floor(forecastDays / 2), total - 1]
    .filter((v, i, a) => a.indexOf(v) === i && v < total);

  return (
    <div className="fc-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block', overflow: 'visible' }}>
        <defs>
          <linearGradient id="hg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#2563eb" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#2563eb" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Y grid + labels */}
        {yTicks.map((v) => {
          const y = sy(v, minV, maxV);
          const label = v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v;
          return (
            <g key={v}>
              <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} stroke="#f1f5f9" strokeWidth="1" />
              <text x={PAD_L - 6} y={y + 4} textAnchor="end" fontSize="10" fill="#94a3b8">{label}</text>
            </g>
          );
        })}

        {/* X labels */}
        {xTickIdxs.map((i) => (
          <text
            key={i}
            x={sx(i, total)}
            y={H - 6}
            textAnchor="middle"
            fontSize="9.5"
            fill="#94a3b8"
          >
            {dateLabel(i - HIST + 1)}
          </text>
        ))}

        {/* Prediction shading */}
        <polygon points={predArea} fill="#eff6ff" opacity="0.75" />

        {/* Historical area */}
        <polygon points={histArea} fill="url(#hg)" />

        {/* Historical line */}
        <polyline
          points={histPts.join(' ')}
          fill="none"
          stroke="#2563eb"
          strokeWidth="2.5"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Prediction line (dashed) */}
        <polyline
          points={predPts.join(' ')}
          fill="none"
          stroke="#60a5fa"
          strokeWidth="2"
          strokeDasharray="7,4"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Divider */}
        <line
          x1={divX} y1={PAD_T}
          x2={divX} y2={bottom}
          stroke="#cbd5e1"
          strokeWidth="1.5"
          strokeDasharray="4,3"
        />
        <rect x={divX - 14} y={PAD_T - 2} width="28" height="14" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="0.8" />
        <text x={divX} y={PAD_T + 8} textAnchor="middle" fontSize="9" fill="#475569" fontWeight="700">오늘</text>

        {/* Axes */}
        <line x1={PAD_L} y1={bottom} x2={W - PAD_R} y2={bottom} stroke="#e2e8f0" />
        <line x1={PAD_L} y1={PAD_T}  x2={PAD_L}     y2={bottom}  stroke="#e2e8f0" />
      </svg>

      {/* Legend */}
      <div className="fc-legend">
        <span className="fc-leg-item">
          <span className="fc-leg-line solid" />
          실제 판매량
        </span>
        <span className="fc-leg-item">
          <span className="fc-leg-line dashed" />
          AI 예측
        </span>
      </div>
    </div>
  );
}
