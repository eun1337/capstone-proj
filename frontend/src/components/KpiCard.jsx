import { useLayoutEffect, useRef, useState } from 'react';
import { Area, AreaChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import './KpiCard.css';

const WEEKDAYS = ['일', '월', '화', '수', '목', '금', '토'];

// x축/호버 날짜 배지용 짧은 표기 — "9/28(토)"
function fmtSparkDate(iso) {
  if (!iso) return '';
  const d = new Date(`${iso}T00:00:00`);
  return `${d.getMonth() + 1}/${d.getDate()}(${WEEKDAYS[d.getDay()]})`;
}

// x축 양끝 고정 라벨용 더 짧은 표기 — "9/1"
function fmtAxisEdge(iso) {
  if (!iso) return '';
  const d = new Date(`${iso}T00:00:00`);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function mean(vals) {
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

// Recharts Area의 dot 렌더 함수 — 전 구간 균일한 작은 점(선과 동일 색)을 찍고, 마지막
// (최신/기준일) 포인트만 반투명 후광 + 속이 빈 링으로 강조한다. XAxis에 좌우 padding을
// 줘서(끝 지점이 plot 경계에 딱 붙지 않도록) 이 후광 반경이 Recharts의 내부 dot clip에
// 잘리지 않게 했다.
function makeDotRenderer(color, lastIndex) {
  return function Dot(props) {
    const { cx, cy, index } = props;
    if (cx == null || cy == null) return null;
    if (index === lastIndex) {
      return (
        <g>
          <circle cx={cx} cy={cy} r={9.5} fill={color} fillOpacity={0.18} />
          <circle cx={cx} cy={cy} r={4.5} fill="#fff" stroke={color} strokeWidth={2} />
        </g>
      );
    }
    return <circle cx={cx} cy={cy} r={2.2} fill={color} stroke="none" />;
  };
}

// 차트 여백(Recharts margin)과 XAxis padding — Recharts 설정값과 우리가 직접 계산하는
// 인덱스↔픽셀 매핑이 항상 같은 수식을 쓰도록 상수로 고정해 공유한다.
const CHART_H = 44;
const M_TOP = 8, M_RIGHT = 2, M_BOTTOM = 2, M_LEFT = 2;
const AXIS_PAD_L = 6, AXIS_PAD_R = 14;
const PLOT_X0 = M_LEFT + AXIS_PAD_L;
const PLOT_X_INSET = M_LEFT + AXIS_PAD_L + M_RIGHT + AXIS_PAD_R;
const PLOT_Y0 = M_TOP;
const PLOT_H = CHART_H - M_TOP - M_BOTTOM;

// ── Mini Area Chart (Sparkline, Recharts AreaChart 기반) ──────────────────
// 단일 연속 곡선(과거/최근 구간 색 분리 없음) + 세로 그라데이션 + 균일한 선 두께 +
// 전 포인트 점 + 마지막 포인트 강조 링. 평균 기준선은 ReferenceLine으로 그린다.
//
// 호버 인터랙션(수치 pill 위 / 날짜 배지 아래 x축 / 크로스헤어 / 하이라이트 서클)은 Recharts의
// 내부 tooltip-active state(activeCoordinate 등)에 기대지 않고, 마우스 좌표에서 우리가 직접
// index↔px를 계산해 그린다 — Recharts 메이저 버전이 바뀌어도(v2→v3 등) 항상 동작하도록 한 것.
// <Tooltip cursor=.../>는 그대로 유지해 Recharts 자체 크로스헤어도 보조적으로 함께 뜬다.
// data: [{ v: number, date?: string }, ...] 또는 number[] (하위 호환)
function Sparkline({ data, color, formatValue }) {
  const wrapRef = useRef(null);
  const valuePillRef = useRef(null);
  const datePillRef = useRef(null);
  const [hoverIdx, setHoverIdx] = useState(null);
  // 두 pill은 기본적으로 hoverX에 중앙 정렬(translateX(-50%))되지만, 텍스트 길이가 길어서
  // (예: ₩56,306,300) 카드 가장자리를 벗어날 것 같으면 실제 렌더된 폭을 측정해 그만큼만
  // 안쪽으로 밀어 넣는다 — 값이 몇 자리든 절대 카드 밖으로 삐져나오지 않는다.
  const [valueShift, setValueShift] = useState(0);
  const [dateShift, setDateShift] = useState(0);

  const hasData = Boolean(data && data.length >= 2);
  const normalized = hasData
    ? data.map((d, i) => (typeof d === 'object' ? { ...d, idx: i } : { v: d, idx: i }))
    : [];
  const n = normalized.length;
  const vals = normalized.map((d) => d.v);
  const avg = mean(vals);
  const lastIndex = n - 1;
  const valMin = hasData ? Math.min(...vals) : 0;
  const valRange = hasData ? (Math.max(...vals) - valMin || 1) : 1;

  const uid = color.replace('#', '');
  const gradId = `kpi-grad-${uid}`;

  const dotRenderer = makeDotRenderer(color, lastIndex);

  function indexToX(idx, width) {
    const plotW = Math.max(1, width - PLOT_X_INSET);
    return PLOT_X0 + (lastIndex > 0 ? (idx / lastIndex) * plotW : 0);
  }
  function indexToY(idx) {
    const v = normalized[idx].v;
    return PLOT_Y0 + (1 - (v - valMin) / valRange) * PLOT_H;
  }

  function handleMove(e) {
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const plotW = Math.max(1, rect.width - PLOT_X_INSET);
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left - PLOT_X0) / plotW));
    setHoverIdx(Math.round(frac * lastIndex));
  }

  const hoverPt = hasData && hoverIdx !== null ? normalized[hoverIdx] : null;
  const wrapWidth = wrapRef.current?.offsetWidth || 0;
  const hoverX = hoverPt ? indexToX(hoverIdx, wrapWidth) : 0;
  const hoverY = hoverPt ? indexToY(hoverIdx) : 0;

  // hoverX를 중심으로 각 pill을 렌더한 뒤 실제 폭을 재서, 카드 좌/우 경계를 넘어가는 만큼만
  // 반대 방향으로 밀어 넣는다(margin 4px). 텍스트가 몇 자리든 항상 카드 안에 들어온다.
  useLayoutEffect(() => {
    if (!hoverPt || !wrapWidth) {
      setValueShift(0);
      setDateShift(0);
      return;
    }
    const margin = 4;
    function clampShift(el) {
      if (!el) return 0;
      const w = el.offsetWidth;
      const left = hoverX - w / 2;
      const right = hoverX + w / 2;
      if (left < margin) return margin - left;
      if (right > wrapWidth - margin) return (wrapWidth - margin) - right;
      return 0;
    }
    setValueShift(clampShift(valuePillRef.current));
    setDateShift(clampShift(datePillRef.current));
  }, [hoverPt, hoverX, wrapWidth]);

  if (!hasData) return null;

  return (
    <div className="kpi-spark-outer">
      <div className="kpi-spark-wrap" ref={wrapRef} onMouseMove={handleMove} onMouseLeave={() => setHoverIdx(null)}>
        <ResponsiveContainer width="100%" height={CHART_H}>
          <AreaChart data={normalized} margin={{ top: M_TOP, right: M_RIGHT, bottom: M_BOTTOM, left: M_LEFT }}>
            <defs>
              <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.2} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>

            {/* x축은 시각적으로는 숨기되(hide), 좌우 padding으로 마지막 포인트의 후광이
                Recharts 내부 dot clipPath에 잘리지 않을 여유를 확보한다. */}
            <XAxis dataKey="idx" type="number" domain={[0, lastIndex]} hide padding={{ left: AXIS_PAD_L, right: AXIS_PAD_R }} />
            <YAxis hide domain={['dataMin', 'dataMax']} />

            {/* 평균 기준선 — 텍스트 없이 은은한 회색 점선만. 숫자는 카드 상단 "일평균"
                서브텍스트로 별도 표시한다. */}
            <ReferenceLine y={avg} stroke="#cbd5e1" strokeDasharray="3 3" strokeWidth={1} ifOverflow="extendDomain" />

            {/* Recharts 자체 크로스헤어(있으면 보조로 같이 뜸) — 실제 값/날짜 표시와 호버
                감지 자체는 위 handleMove가 독립적으로 책임진다(Recharts 내부 상태에 기대지 않음). */}
            <Tooltip
              content={() => null}
              cursor={{ stroke: '#94a3b8', strokeWidth: 1, strokeDasharray: '3 3' }}
              isAnimationActive={false}
            />

            <Area
              type="monotone"
              dataKey="v"
              stroke={color}
              strokeWidth={2}
              fill={`url(#${gradId})`}
              dot={dotRenderer}
              activeDot={{ r: 3, fill: '#fff', stroke: color, strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>

        {/* 아래 3개(크로스헤어/하이라이트 서클/값 pill)는 전부 handleMove가 계산한 hoverX/hoverY로
            그린다 — Recharts 렌더링과 별개의 독립된 오버레이라 항상 즉각적으로 반응한다. */}
        {hoverPt && <div className="kpi-spark-crosshair" style={{ left: `${hoverX}px` }} />}
        {hoverPt && (
          <div className="kpi-spark-hover-dot" style={{ left: `${hoverX}px`, top: `${hoverY}px`, borderColor: color }} />
        )}

        {/* 값 pill — 차트 위에 숫자만 단독으로. 날짜는 여기 없다(아래 x축으로 완전히 분리).
            기본은 hoverX 중앙 정렬이고, useLayoutEffect가 잰 valueShift만큼 안쪽으로 보정된다. */}
        {hoverPt && (
          <div
            ref={valuePillRef}
            className="kpi-spark-value-pill"
            style={{ left: `${hoverX}px`, transform: `translateX(calc(-50% + ${valueShift}px))` }}
          >
            {formatValue ? formatValue(hoverPt.v) : Math.round(hoverPt.v).toLocaleString()}
          </div>
        )}
      </div>

      {/* x축 — 평소엔 양 끝(시작/기준일) 날짜만 옅게, 호버 중엔 그 지점 날짜가 강조 배지로 뜬다 */}
      <div className="kpi-spark-axis">
        <span className="kpi-spark-axis-edge">{fmtAxisEdge(normalized[0]?.date)}</span>
        <span className="kpi-spark-axis-edge">{fmtAxisEdge(normalized[lastIndex]?.date)}</span>
        {hoverPt && (
          <div
            ref={datePillRef}
            className="kpi-spark-date-pill"
            style={{ left: `${hoverX}px`, transform: `translateY(-50%) translateX(calc(-50% + ${dateShift}px))` }}
          >
            {fmtSparkDate(hoverPt.date)}
          </div>
        )}
      </div>
    </div>
  );
}

// ── KPI Card ───────────────────────────────────────────────────
// - changePct/changeAbsText: "전일 대비"(DoD) — 값 아래 보조 텍스트로만 표시.
// - wowPct: "전주 대비"(WoW, 기준일 vs 기준일-7일 동일 요일) — 메인 수치 옆 배지로 표시.
// - sparkRangeDays(7|30) + onToggleSparkRange: 카드 우측 상단 기간 토글. 3개 카드가 같은
//   state를 공유하도록 부모가 내려준다(한 카드에서 토글하면 3개 전부 동기화).
// - sparkData: { v, date? }[] — 이미 sparkRangeDays만큼 잘려서 내려온다. 이 배열의
//   평균을 "일평균" 서브텍스트로 표시한다(차트 안에는 숫자를 넣지 않는다 — 대안 A).
// onClick이 주어지면(대시보드 KPI 3개) 카드 전체가 클릭 가능한 설명 modal 트리거가 된다.
export default function KpiCard({
  label, value, changePct, changeAbsText,
  wowPct, sparkData, sparkValueFormat, color,
  sparkRangeDays, onToggleSparkRange, onClick,
}) {
  const hasChange = changePct !== null && changePct !== undefined;
  const hasWow = wowPct !== null && wowPct !== undefined;
  const isWowUp = hasWow && wowPct >= 0;
  const hasSpark = Boolean(sparkData && sparkData.length >= 2);
  const hasRangeToggle = Boolean(onToggleSparkRange);
  const avgValue = hasSpark ? mean(sparkData.map((d) => d.v)) : null;

  const content = (
    <>
      <div className="kpi-top">
        <p className="kpi-label">{label}</p>
        {hasRangeToggle && (
          <button
            type="button"
            className="kpi-range-btn"
            onClick={(e) => { e.stopPropagation(); onToggleSparkRange(); }}
            title="스파크라인 조회 기간 전환"
          >
            {sparkRangeDays === 7 ? '30일' : '7일'}
          </button>
        )}
      </div>

      <div className="kpi-value-row">
        <span className="kpi-value">{value}</span>
        {hasWow && (
          <span className="kpi-wow">
            <span className="kpi-wow-label">전주 대비</span>
            <span className={`kpi-badge ${isWowUp ? 'badge-up' : 'badge-dn'}`}>
              {isWowUp ? '▲' : '▼'} {Math.abs(wowPct).toFixed(1)}%
            </span>
          </span>
        )}
      </div>

      {avgValue !== null && (
        <p className="kpi-avg-row">
          일평균 {sparkValueFormat ? sparkValueFormat(avgValue) : Math.round(avgValue).toLocaleString()}
        </p>
      )}

      {hasChange && changeAbsText && (
        <p className="kpi-change-row">
          <span className="kpi-change-label">전일 대비</span>
          <span className="kpi-change-abs">{changeAbsText}</span>
        </p>
      )}

      {hasSpark && (
        <div className="kpi-spark">
          <Sparkline data={sparkData} color={color || '#3b82f6'} formatValue={sparkValueFormat} />
        </div>
      )}
    </>
  );

  // onClick이 있으면 카드 전체가 클릭 가능해지는데, 내부에 기간 토글 <button>이 함께 들어가므로
  // <button> 중첩(불가) 대신 InsightCard와 동일하게 role="button" div로 감싼다.
  if (onClick) {
    return (
      <div
        className="kpi-card kpi-card-clickable"
        role="button"
        tabIndex={0}
        onClick={onClick}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick(); }}
        title={`${label} 자세히 보기`}
      >
        {content}
      </div>
    );
  }

  return <div className="kpi-card">{content}</div>;
}
