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

  const first = `0,${H}`;
  const last  = `${W},${H}`;
  const area  = `${first} ${pts} ${last}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
      <defs>
        <linearGradient id={`sg-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon points={area} fill={`url(#sg-${color.replace('#', '')})`} />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// ── KPI Card ───────────────────────────────────────────────────
// 전일 대비는 반드시 "증감률 → 증감액/증감건수/증감SKU수" 순서로 한 줄에 표시한다.
// onClick이 주어지면(대시보드 KPI 3개) 카드 전체가 클릭 가능한 설명 modal 트리거가 된다.
export default function KpiCard({ label, icon, iconColor, value, changePct, changeAbsText, sparkData, color, onClick }) {
  const hasChange = changePct !== null && changePct !== undefined;
  const isUp = hasChange && changePct >= 0;
  const hasSpark = Boolean(sparkData && sparkData.length >= 2);

  const content = (
    <>
      <div className="kpi-top">
        <p className="kpi-label">{label}</p>
        {icon && (
          <span className="kpi-icon-circle" style={{ background: `${iconColor}1a`, color: iconColor }}>
            {icon}
          </span>
        )}
      </div>

      <div className="kpi-value">{value}</div>

      {hasChange && (
        <p className="kpi-change-row">
          <span className="kpi-change-label">전일 대비</span>
          <span className={`kpi-change-pct ${isUp ? 'kpi-up' : 'kpi-dn'}`}>
            {isUp ? '▲' : '▼'} {Math.abs(changePct).toFixed(2)}%
          </span>
          {changeAbsText && <span className="kpi-change-abs">{changeAbsText}</span>}
        </p>
      )}

      <div className="kpi-spark">
        <Sparkline data={sparkData} color={color} />
      </div>
    </>
  );

  if (onClick) {
    return (
      <button type="button" className="kpi-card kpi-card-clickable" onClick={onClick} title={`${label} 자세히 보기`}>
        {content}
      </button>
    );
  }

  return <div className="kpi-card">{content}</div>;
}
