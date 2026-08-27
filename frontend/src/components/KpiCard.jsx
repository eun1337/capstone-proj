import './KpiCard.css';

// ── Mini Sparkline ─────────────────────────────────────────────
function Sparkline({ data, color }) {
  if (!data || data.length < 2) return null;
  const W = 120;
  const H = 38;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const rng = max - min || 1;

  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * W;
      const y = H - 4 - ((v - min) / rng) * (H - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  // Area fill path
  const first = `0,${H}`;
  const last  = `${W},${H}`;
  const area  = `${first} ${pts} ${last}`;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      height={H}
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id={`sg-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon
        points={area}
        fill={`url(#sg-${color.replace('#', '')})`}
      />
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.8"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

// ── KPI Card ───────────────────────────────────────────────────
export default function KpiCard({ label, value, change, sparkData, color }) {
  const isUp = change >= 0;

  return (
    <div className="kpi-card">
      <div className="kpi-top">
        <p className="kpi-label">{label}</p>
        <span className={`kpi-badge ${isUp ? 'badge-up' : 'badge-dn'}`}>
          {isUp ? '▲' : '▼'}&nbsp;{Math.abs(change).toFixed(1)}%
        </span>
      </div>

      <div className="kpi-value">{value}</div>

      <p className="kpi-pw">vs. PW</p>

      <div className="kpi-spark">
        <Sparkline data={sparkData} color={color} />
      </div>
    </div>
  );
}
