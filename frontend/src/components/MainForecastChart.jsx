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

// ISO 8601 주차 라벨 — 기준일(항상 월요일, week_st)을 "'24년 40W" 형태로 표기한다.
// (2024-09-30 → 40주차로 검증됨: 그 해 1/1이 월요일이라 1주=1/1~1/7)
function isoWeekLabel(dateStr) {
  if (!dateStr) return '';
  const d = new Date(`${dateStr}T00:00:00`);
  const dayNum = (d.getDay() + 6) % 7; // 월=0 ... 일=6
  d.setDate(d.getDate() - dayNum + 3); // 그 주의 목요일로 이동(ISO 주차는 목요일 기준)
  const firstThursday = new Date(d.getFullYear(), 0, 4);
  const fDayNum = (firstThursday.getDay() + 6) % 7;
  firstThursday.setDate(firstThursday.getDate() - fDayNum + 3);
  const week = 1 + Math.round((d - firstThursday) / (7 * DAY_MS));
  const yy = String(d.getFullYear()).slice(-2);
  return `${yy}년 ${week}W`;
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
    if (allPoints.length === 0) return { histPoints, predPoints, minV: 0, maxV: 1 };

    const vals = allPoints.map((p) => p.v);
    const minV = Math.min(...vals, 0);
    const maxV = (Math.max(...vals) || 0) * 1.15 || 1;

    return { histPoints, predPoints, minV, maxV };
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

  const { histPoints, predPoints, minV, maxV } = chart;
  const bottom = PAD_T + EH;
  const lastHist = histPoints[histPoints.length - 1];

  // 실적 구간(왼쪽 PAST_SHARE)과 예측 구간(오른쪽 나머지)을 각각 독립된 선형 스케일로 배치한다.
  // 각 구간 안에서는 실제 날짜 차이에 정확히 비례하지만(진짜 time scale), 두 구간의 폭 배분
  // 자체는 고정 비율이라 예측 몇 개월/조회기간과 무관하게 항상 같은 자리(대략 80% 지점)에
  // 기준선이 온다.
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
        <rect x={divX - 26} y={PAD_T - 2} width="52" height="15" rx="4" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="0.8" />
        <text x={divX} y={PAD_T + 9} textAnchor="middle" fontSize="9" fill="#475569" fontWeight="700">
          {basisWeekLabel}
        </text>

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
          예측 수요
        </span>
      </div>
    </div>
  );
}
