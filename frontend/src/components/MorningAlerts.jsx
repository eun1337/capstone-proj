import './MorningAlerts.css';

// 모닝 긴급 알림 — 출근 직후 오늘/내일 조치가 필요한 3~4건만 한 줄씩 보여준다.
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

// onAction(type): 결품 계열(D_DAY/NEXT_DAY)은 결품 위험 상세, SPIKE는 판매 급증 상세를 연다.
export default function MorningAlerts({ alerts, onAction }) {
  if (!alerts || alerts.length === 0) return null;
  return (
    <div className="ma-strip">
      <div className="ma-label">모닝 긴급 알림</div>
      <div className="ma-list">
        {alerts.map((a) => (
          <div key={`${a.type}-${a.title}`} className={`ma-item ma-${a.type.toLowerCase()}`}>
            <span className="ma-badge">{a.badge}</span>
            <span className="ma-title" title={a.title}>{a.title}</span>
            <span className="ma-summary">{a.summary}</span>
            <button type="button" className="ma-action" onClick={() => onAction?.(a.type)}>
              {a.action}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
