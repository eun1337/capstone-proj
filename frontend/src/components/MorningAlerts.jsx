import { useEffect, useRef, useState } from 'react';
import './MorningAlerts.css';

// 모닝 긴급 알림 — 출근 직후 오늘/내일 조치가 필요한 3~4건을 헤더 🔔 드롭다운으로 보여준다.
// 재고 데이터가 mock이라 아직 API가 없어, 시연 기준(A센터 · 2024-10-04) 결과를 고정값으로 둔다.
// 코카콜라 355ml 캔(잔여 1EA)은 최대 4건 규칙상 태그별 1건 이상을 남기려고 제외했다.
const DEMO_ALERTS = {
  'A|2024-10-04': [
    { type: 'D_DAY', badge: '🚨 금일 소진', title: '트레비 자몽 500ml', summary: '0EA / 예측치 미제공', action: '긴급 발주' },
    { type: 'D_DAY', badge: '🚨 금일 소진', title: '갈배 사이다 500ml', summary: '0EA / 예측치 미제공', action: '긴급 발주' },
    { type: 'NEXT_DAY', badge: '⚠️ 명일 소진', title: '코카콜라 500ml', summary: '64EA / 184EA', action: '보충 발주' },
    { type: 'SPIKE', badge: '📈 수요 급증', title: '칠성사이다 190ml 업소용', summary: '전주 대비 +120EA', action: '재고 확보' },
  ],
};

export function getMorningAlerts(center, operationalDate) {
  return DEMO_ALERTS[`${center}|${operationalDate}`] || [];
}

// 헤더 우측 알림 버튼 + 드롭다운. 알림이 없어도 버튼은 남기고 뱃지만 숨긴다.
// onAction(type): 결품 계열(D_DAY/NEXT_DAY)은 결품 위험 상세, SPIKE는 판매 급증 상세를 연다.
export default function MorningAlertBell({ alerts, onAction }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const count = alerts?.length ?? 0;

  // 드롭다운 바깥 클릭 / Esc 시 닫는다.
  useEffect(() => {
    if (!open) return undefined;
    function handlePointerDown(e) {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    }
    function handleKeyDown(e) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [open]);

  return (
    <div className="ma-bell-root" ref={rootRef}>
      <button
        type="button"
        className={`ma-bell ${open ? 'open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        aria-label={`알림 ${count}건`}
        aria-expanded={open}
        title="알림"
      >
        <span aria-hidden="true">🔔</span>
        {count > 0 && <span className="ma-bell-badge">{count}</span>}
      </button>

      {open && (
        <div className="ma-popover" role="dialog" aria-label="알림">
          <div className="ma-popover-head">
            <span>알림</span>
            <span className="ma-popover-count">{count}건</span>
          </div>
          {count === 0 ? (
            <div className="ma-empty">오늘 조치가 필요한 알림이 없습니다.</div>
          ) : (
            <ul className="ma-list">
              {alerts.map((a) => (
                <li key={`${a.type}-${a.title}`} className={`ma-item ma-${a.type.toLowerCase()}`}>
                  <div className="ma-text">
                    <span className="ma-badge">{a.badge}</span>
                    <span className="ma-title" title={a.title}>{a.title}</span>
                    <span className="ma-summary">{a.summary}</span>
                  </div>
                  <button
                    type="button"
                    className="ma-action"
                    onClick={() => { setOpen(false); onAction?.(a.type); }}
                  >
                    {a.action}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
