import { useLayoutEffect, useMemo, useRef, useState } from 'react';
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

// 과거 실적 구간이 차트 폭에서 차지하는 비율(나머지는 예측 3포인트 구간).
// 실적/예측 각각의 "구간 내부" 배치는 실제 경과일에 비례(진짜 time scale)하지만, 두 구간이
// 전체 폭에서 나눠 갖는 몫 자체는 조회기간(historyWeeks)과 무관하게 고정한다 — 그래야
// 짧은 기간을 조회해도(예: 7일/30일) 예측 4주 구간이 과거 구간만큼 넓어지는 일이 없다.
const PAST_SHARE = 0.8;

function linScale(dateStr, minDate, maxDate, x0, width) {
  const span = (new Date(maxDate) - new Date(minDate)) / DAY_MS || 1;
  const off  = (new Date(dateStr) - new Date(minDate)) / DAY_MS;
  return x0 + (off / span) * width;
}

const sy = (v, min, max) => {
  const rng = max - min || 1;
  return PAD_T + (1 - (v - min) / rng) * EH;
};

function shortDate(dateStr) {
  const d = new Date(dateStr);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function mmdd(d) {
  return `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// ISO 주차 계산 — 기준일(항상 월요일, week_st)을 그 주의 목요일로 이동시켜(ISO 주차는
// 목요일 기준) 연도/주차를 구한다. (2024-09-30 → 40주차로 검증됨: 그 해 1/1이 월요일이라
// 1주=1/1~1/7)
function isoWeekParts(dateStr) {
  const d = new Date(`${dateStr}T00:00:00`);
  const dayNum = (d.getDay() + 6) % 7; // 월=0 ... 일=6
  d.setDate(d.getDate() - dayNum + 3);
  const firstThursday = new Date(d.getFullYear(), 0, 4);
  const fDayNum = (firstThursday.getDay() + 6) % 7;
  firstThursday.setDate(firstThursday.getDate() - fDayNum + 3);
  const week = 1 + Math.round((d - firstThursday) / (7 * DAY_MS));
  return { year: d.getFullYear(), week };
}

// x축 divider 배지용 — "'24년 40W"
function isoWeekLabel(dateStr) {
  if (!dateStr) return '';
  const { year, week } = isoWeekParts(dateStr);
  return `${String(year).slice(-2)}년 ${week}W`;
}

// 실적 구간 호버 툴팁용 — "2024년 38W (09-16 ~ 09-22)" (week_st가 그 주 월요일이므로 +6일이 일요일)
function histWeekRangeLabel(weekStart) {
  const { year, week } = isoWeekParts(weekStart);
  const start = new Date(`${weekStart}T00:00:00`);
  const end = new Date(start);
  end.setDate(end.getDate() + 6);
  return `${year}년 ${week}W (${mmdd(start)} ~ ${mmdd(end)})`;
}

// 예측 구간 호버 툴팁용 — "24년 41W (+1주 후 예측치) · 2024-10-07"
function predWeekLabel(point) {
  return `${isoWeekLabel(point.date)} (+${point.dateLabel} 예측치) · ${point.date}`;
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
    }));

    const predPoints = horizons.map((h) => ({
      date: h.target_date,
      v: h.predicted_qty,
      dateLabel: h.label,
      isPred: true,
    }));

    const allPoints = [...histPoints, ...predPoints];
    if (allPoints.length === 0) return { histPoints, predPoints, minV: 0, maxV: 1 };

    const vals = allPoints.map((p) => p.v);
    const minV = Math.min(...vals, 0);
    const maxV = (Math.max(...vals) || 0) * 1.15 || 1;

    return { histPoints, predPoints, minV, maxV };
  }, [trend]);

  // 실적 구간(왼쪽 PAST_SHARE)과 예측 구간(오른쪽 나머지)을 각각 독립된 선형 스케일로 배치하는
  // 좌표 변환 + 호버 판정용 포인트 목록. loading/error/데이터없음일 때도 chart가 null일 수
  // 있어 아래 훅들과 분리해서 항상 안전하게(null 허용) 계산해 둔다.
  const geometry = useMemo(() => {
    if (!chart || chart.histPoints.length === 0) return null;
    const { histPoints, predPoints, minV, maxV } = chart;
    const bottom = PAD_T + EH;
    const lastHist = histPoints[histPoints.length - 1];
    const hasPred = predPoints.length > 0;
    const pastEW = hasPred ? EW * PAST_SHARE : EW;
    const futureEW = EW - pastEW;
    const divX = PAD_L + pastEW;
    const histMinDate = histPoints[0].date;
    const histMaxDate = lastHist.date;
    const futureMaxDate = hasPred ? predPoints[predPoints.length - 1].date : histMaxDate;

    const pxHist = (dateStr) => linScale(dateStr, histMinDate, histMaxDate, PAD_L, pastEW);
    const pxPred = (dateStr) => (hasPred ? linScale(dateStr, histMaxDate, futureMaxDate, divX, futureEW) : divX);
    const px = (p) => (p.isPred ? pxPred(p.date) : pxHist(p.date));

    const combined = [
      ...histPoints.map((p, i) => ({ ...p, uid: `h-${i}` })),
      ...predPoints.map((p, i) => ({ ...p, uid: `p-${i}` })),
    ];

    return { bottom, lastHist, hasPred, divX, px, minV, maxV, combined };
  }, [chart]);

  // ── 호버 인터랙션 상태 ──────────────────────────────────────────
  // KpiCard 스파크라인과 같은 방식: Recharts/차트 라이브러리 상태에 기대지 않고, 마우스
  // 좌표에서 직접 가장 가까운 데이터 포인트(uid)를 찾아 크로스헤어 + 커스텀 툴팁을 그린다.
  const wrapRef = useRef(null);
  const tooltipRef = useRef(null);
  const [hoverUid, setHoverUid] = useState(null);
  const [tipShiftX, setTipShiftX] = useState(0);
  const [tipAbove, setTipAbove] = useState(true);

  const hoverPoint = geometry && hoverUid
    ? geometry.combined.find((p) => p.uid === hoverUid) ?? null
    : null;
  const hoverX = hoverPoint ? geometry.px(hoverPoint) : 0;
  const hoverY = hoverPoint ? sy(hoverPoint.v, geometry.minV, geometry.maxV) : 0;

  // 툴팁이 카드 좌/우 경계를 넘어가면(양 끝 포인트 호버 시) 안쪽으로 밀어 넣고, 위쪽 여유가
  // 부족하면(최댓값 근처 포인트) 포인트 아래쪽으로 뒤집어서 — 조회기간(2주/30일/90일)이
  // 무엇이든, 어느 포인트를 호버하든 툴팁이 차트 밖으로 잘리지 않게 한다.
  useLayoutEffect(() => {
    if (!hoverPoint || !wrapRef.current || !tooltipRef.current) {
      setTipShiftX(0);
      setTipAbove(true);
      return;
    }
    const wrapWidth = wrapRef.current.offsetWidth;
    const scale = wrapWidth / W;
    const hoverXPx = hoverX * scale;
    const hoverYPx = hoverY * scale;
    const tw = tooltipRef.current.offsetWidth;
    const th = tooltipRef.current.offsetHeight;
    const margin = 4;
    let shift = 0;
    const left = hoverXPx - tw / 2;
    const right = hoverXPx + tw / 2;
    if (left < margin) shift = margin - left;
    else if (right > wrapWidth - margin) shift = (wrapWidth - margin) - right;
    setTipShiftX(shift);
    setTipAbove(hoverYPx - th - 10 >= 0);
  }, [hoverPoint, hoverX, hoverY]);

  function handleChartMouseMove(e) {
    if (!geometry || !wrapRef.current) return;
    const rect = wrapRef.current.getBoundingClientRect();
    if (!rect.width) return;
    const vbX = ((e.clientX - rect.left) / rect.width) * W;
    let nearest = null;
    let nearestDist = Infinity;
    for (const p of geometry.combined) {
      const d = Math.abs(geometry.px(p) - vbX);
      if (d < nearestDist) { nearestDist = d; nearest = p; }
    }
    setHoverUid(nearest ? nearest.uid : null);
  }

  function handleChartMouseLeave() {
    setHoverUid(null);
  }

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

  const { histPoints, predPoints, minV, maxV } = chart;
  const { bottom, lastHist, divX, px } = geometry;

  const xy = (p) => `${px(p).toFixed(1)},${sy(p.v, minV, maxV).toFixed(1)}`;

  const histLine = histPoints.map(xy).join(' ');
  const histArea = [...histPoints.map(xy), `${divX.toFixed(1)},${bottom}`, `${PAD_L},${bottom}`].join(' ');

  // 예측 구간은 실적과 명확히 분리하기 위해 영역 채우기(fill) 없이 점선 + 포인트 링만 그린다.
  const predLinePts = predPoints.length ? [lastHist, ...predPoints].map(xy).join(' ') : '';
  const basisWeekLabel = isoWeekLabel(lastHist.date);

  // 조회기간(historyWeeks)이 늘어날수록 histPoints가 촘촘해져 x축 날짜 라벨이 겹치므로,
  // 실제 픽셀 간격을 보고 최소 간격(MIN_LABEL_GAP_PX)이 나오는 배수로만 골라서 그린다.
  // 양 끝(조회 범위의 첫/마지막 실적 포인트)은 배수와 무관하게 항상 표시한다.
  const histSpacingPx = histPoints.length > 1
    ? (px(histPoints[histPoints.length - 1]) - px(histPoints[0])) / (histPoints.length - 1)
    : EW;
  const MIN_LABEL_GAP_PX = 26;
  const histLabelStep = Math.max(1, Math.ceil(MIN_LABEL_GAP_PX / Math.max(histSpacingPx, 0.01)));

  const yTickCount = 4;
  const yStep = (maxV - minV) / yTickCount || 1;
  const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => Math.round(minV + i * yStep));

  return (
    <div className="mfc-wrap">
      <div
        ref={wrapRef}
        className="mfc-chart-area"
        onMouseMove={handleChartMouseMove}
        onMouseLeave={handleChartMouseLeave}
      >
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

          {histPoints.map((p, idx) => {
            const isEdge = idx === 0 || idx === histPoints.length - 1;
            if (!isEdge && idx % histLabelStep !== 0) return null;
            return (
              <text key={`h-${idx}`} x={px(p)} y={H - 8} textAnchor="middle" fontSize="9.5" fontWeight="400" fill="#94a3b8">
                {p.dateLabel}
              </text>
            );
          })}
          {predPoints.map((p, idx) => (
            <text key={`p-${idx}`} x={px(p)} y={H - 8} textAnchor="middle" fontSize="10" fontWeight="700" fill="#2563eb">
              {p.dateLabel}
            </text>
          ))}

          {/* 호버 크로스헤어 — 실제 값/포인트 감지는 위 handleChartMouseMove가 독립적으로
              계산하고, 이 선은 그 결과(hoverX)를 그대로 그리는 오버레이일 뿐이다. */}
          {hoverPoint && (
            <line x1={hoverX} y1={PAD_T} x2={hoverX} y2={bottom} stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 3" pointerEvents="none" />
          )}

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

          {histPoints.map((p, idx) => {
            const uid = `h-${idx}`;
            return (
              <circle key={uid} cx={px(p)} cy={sy(p.v, minV, maxV)} r={hoverUid === uid ? 5 : 2.8} fill="#2563eb" />
            );
          })}
          {predPoints.map((p, idx) => {
            const uid = `p-${idx}`;
            return (
              <circle key={uid} cx={px(p)} cy={sy(p.v, minV, maxV)} r={hoverUid === uid ? 5 : 4} fill="#fff" stroke="#60a5fa" strokeWidth="2" />
            );
          })}

          <line x1={divX} y1={PAD_T} x2={divX} y2={bottom} stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="4,3" />
          <rect x={divX - 26} y={PAD_T - 2} width="52" height="15" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="0.8" />
          <text x={divX} y={PAD_T + 9} textAnchor="middle" fontSize="9" fill="#475569" fontWeight="700">
            {basisWeekLabel}
          </text>

          <line x1={PAD_L} y1={bottom} x2={W - PAD_R} y2={bottom} stroke="#e2e8f0" />
          <line x1={PAD_L} y1={PAD_T}  x2={PAD_L}     y2={bottom}  stroke="#e2e8f0" />
        </svg>

        {/* 커스텀 툴팁 — hoverX/hoverY(viewBox 좌표)를 %로 배치한 뒤, 측정된 실제 폭/높이
            기준으로 좌우/상하 클램프(tipShiftX/tipAbove)만 transform에 보정해 넣는다. */}
        {hoverPoint && (
          <div
            ref={tooltipRef}
            className="mfc-tooltip"
            style={{
              left: `${(hoverX / W) * 100}%`,
              top: `${(hoverY / H) * 100}%`,
              transform: `translate(calc(-50% + ${tipShiftX}px), ${tipAbove ? 'calc(-100% - 10px)' : '10px'})`,
            }}
          >
            <div className="mfc-tooltip-week">
              {hoverPoint.isPred ? predWeekLabel(hoverPoint) : histWeekRangeLabel(hoverPoint.date)}
            </div>
            <div className="mfc-tooltip-value">
              <span className={`mfc-tooltip-dot ${hoverPoint.isPred ? 'dashed' : 'solid'}`} />
              {hoverPoint.isPred ? '예측 수요' : '실제 판매수량'}: {Math.round(hoverPoint.v).toLocaleString()} {unit}
            </div>
          </div>
        )}
      </div>

      <div className="mfc-legend">
        <span className="mfc-leg-item">
          <span className="mfc-leg-line solid" />
          실제 판매수량
        </span>
        <span className="mfc-leg-item">
          <span className="mfc-leg-line dashed" />
          예측 수요
        </span>
      </div>
    </div>
  );
}
