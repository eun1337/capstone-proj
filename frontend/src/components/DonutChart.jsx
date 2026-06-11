import './DonutChart.css';

// ── Donut data ─────────────────────────────────────────────────
const SEGMENTS = [
  { label: '높음',  desc: 'High Risk', pct: 15, color: '#ef4444', count: 2 },
  { label: '경고',  desc: 'Warning',   pct: 25, color: '#f59e0b', count: 3 },
  { label: '보통',  desc: 'Normal',    pct: 45, color: '#10b981', count: 6 },
  { label: '낮음',  desc: 'Low Risk',  pct: 15, color: '#3b82f6', count: 2 },
];

const TOTAL_ITEMS = SEGMENTS.reduce((s, seg) => s + seg.count, 0);

// ── SVG Donut ──────────────────────────────────────────────────
const CX   = 76;
const CY   = 76;
const R    = 56;
const SW   = 20; // stroke-width
const CIRC = 2 * Math.PI * R;
const GAP  = 3;  // visual gap between segments (SVG units)

function DonutSegments() {
  let offset = 0;
  return SEGMENTS.map((seg) => {
    const dash = Math.max(0, (seg.pct / 100) * CIRC - GAP);
    const rot  = (offset / 100) * 360 - 90;
    offset += seg.pct;
    return (
      <circle
        key={seg.label}
        cx={CX} cy={CY} r={R}
        fill="none"
        stroke={seg.color}
        strokeWidth={SW}
        strokeDasharray={`${dash.toFixed(2)} ${CIRC.toFixed(2)}`}
        transform={`rotate(${rot.toFixed(2)} ${CX} ${CY})`}
        strokeLinecap="butt"
      />
    );
  });
}

export default function DonutChart() {
  return (
    <div className="donut-wrap">
      <div className="donut-svg-wrap">
        <svg width={CX * 2} height={CY * 2} viewBox={`0 0 ${CX * 2} ${CY * 2}`}>
          {/* Track */}
          <circle cx={CX} cy={CY} r={R} fill="none" stroke="#f1f5f9" strokeWidth={SW} />
          <DonutSegments />
          {/* Center labels */}
          <text x={CX} y={CY - 7} textAnchor="middle" fontSize="22" fontWeight="800" fill="#1e293b">
            {TOTAL_ITEMS}
          </text>
          <text x={CX} y={CY + 11} textAnchor="middle" fontSize="10" fill="#64748b">
            전체 품목
          </text>
        </svg>
      </div>

      <div className="donut-legend">
        {SEGMENTS.map((seg) => {
          const barPct = seg.pct;
          return (
            <div key={seg.label} className="legend-row">
              <span className="legend-dot" style={{ background: seg.color }} />
              <div className="legend-info">
                <div className="legend-top">
                  <span className="legend-name">{seg.label}</span>
                  <span className="legend-desc">{seg.desc}</span>
                </div>
                <div className="legend-bar-wrap">
                  <div className="legend-bar-track">
                    <div
                      className="legend-bar-fill"
                      style={{ width: `${barPct * 2}%`, background: seg.color }}
                    />
                  </div>
                  <span className="legend-meta">
                    <span className="legend-count">{seg.count}개</span>
                    <span className="legend-pct">{seg.pct}%</span>
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
