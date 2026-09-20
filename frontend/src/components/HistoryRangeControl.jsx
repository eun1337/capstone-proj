import { useEffect, useRef, useState } from 'react';
import './HistoryRangeControl.css';

// 2주/30일/90일 프리셋 — 데이터 grain이 주간(week_st)이라 서버는 history_weeks(주 단위)만
// 받으므로, 일 단위 프리셋은 가장 가까운 주 단위로 환산해 둔다(30일→4주, 90일→13주).
// 최소 단위는 2주로 강제한다 — 실적 포인트가 1주(점 1개)뿐이면 추세선이 사실상 안 보여서,
// 최소한 점 2개(=지난 1주간의 변화)는 항상 나오게 한다.
const MIN_WEEKS = 2;

const PRESETS = [
  { label: '2주', weeks: MIN_WEEKS },
  { label: '30일', weeks: 4 },
  { label: '90일', weeks: 13 },
];

function fmtISODate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function shiftDate(iso, days) {
  const d = new Date(`${iso}T00:00:00`);
  d.setDate(d.getDate() + days);
  return fmtISODate(d);
}

// 사용자가 시작일을 직접 고르면(끝은 항상 기준일 고정), 그 일수 차이를 가장 가까운
// 주 단위로 반올림해 history_weeks로 환산한다. 최소 2주는 항상 보장한다.
function weeksFromStart(startIso, endIso) {
  const days = Math.round((new Date(`${endIso}T00:00:00`) - new Date(`${startIso}T00:00:00`)) / 86400000);
  return Math.max(MIN_WEEKS, Math.round((days + 1) / 7));
}

function CalendarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  );
}

function ChevronIcon({ open }) {
  return (
    <svg
      width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
      strokeLinecap="round" strokeLinejoin="round"
      style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

// 수요예측 추이 차트 헤더의 "조회 기간" 컨트롤.
// 좌측: 날짜범위 pill(클릭 시 시작일을 직접 고르는 작은 팝오버가 뜬다 — 끝(기준일)은 고정).
// 우측: 2주/30일/90일 프리셋 버튼. historyWeeks가 프리셋의 주(week) 값과 정확히 일치할
// 때만 그 버튼이 active로 표시되므로, 직접 선택으로 어긋나면 자동으로 전부 해제된다.
export default function HistoryRangeControl({ operationalDate, historyWeeks, onChangeWeeks }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    function handleOutside(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener('mousedown', handleOutside);
    return () => document.removeEventListener('mousedown', handleOutside);
  }, [open]);

  const startDate = shiftDate(operationalDate, -(historyWeeks * 7 - 1));
  const activePreset = PRESETS.find((p) => p.weeks === historyWeeks)?.label ?? null;

  function handleStartChange(e) {
    const nextStart = e.target.value;
    if (!nextStart) return;
    onChangeWeeks(weeksFromStart(nextStart, operationalDate));
  }

  return (
    <div className="hrc-wrap" ref={wrapRef}>
      <button type="button" className="hrc-pill" onClick={() => setOpen((v) => !v)}>
        <CalendarIcon />
        <span className="hrc-pill-label">조회 기간</span>
        <span className="hrc-pill-value">{startDate} ~ {operationalDate}</span>
        <ChevronIcon open={open} />
      </button>

      {open && (
        <div className="hrc-popover">
          <label className="hrc-popover-label" htmlFor="hrc-start-input">시작일</label>
          <input
            id="hrc-start-input"
            type="date"
            className="hrc-popover-input"
            value={startDate}
            max={shiftDate(operationalDate, -(MIN_WEEKS * 7 - 1))}
            onChange={handleStartChange}
          />
          <p className="hrc-popover-hint">종료일은 기준일({operationalDate})로 고정됩니다. 최소 조회 기간은 2주입니다.</p>
        </div>
      )}

      <div className="hrc-presets">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            className={`hrc-preset-btn ${activePreset === p.label ? 'active' : ''}`}
            onClick={() => { onChangeWeeks(p.weeks); setOpen(false); }}
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}
