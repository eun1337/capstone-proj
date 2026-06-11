import './SummaryCard.css';

export default function SummaryCard({ type, summary, onClick }) {
  const isDemand = type === 'demand';

  if (!summary) {
    return <div className="sum-card skeleton" />;
  }

  return (
    <div className="sum-card" onClick={onClick} role="button" tabIndex={0}>
      <div className="sum-card-header">
        <div className="sum-icon">{isDemand ? '📈' : '🏭'}</div>
        <div className="sum-titles">
          <h3>{isDemand ? '수요 예측' : '재고 현황'}</h3>
          <p>{isDemand ? 'Demand Forecasting' : 'Inventory Status'}</p>
        </div>
        <span className="sum-arrow">→</span>
      </div>

      <div className="sum-stats">
        {isDemand ? (
          <>
            <StatItem label="총 예측 수량" value={summary.total_forecast_qty.toLocaleString()} />
            <StatItem label="부족 품목"    value={summary.shortage_items} color="danger" />
            <StatItem label="주의 품목"    value={summary.caution_items}  color="warning" />
          </>
        ) : (
          <>
            <StatItem label="총 재고 수량" value={summary.total_stock_qty.toLocaleString()} />
            <StatItem label="정상 품목"    value={summary.normal_items}   color="success" />
            <StatItem label="부족 품목"    value={summary.shortage_items} color="danger" />
          </>
        )}
      </div>

      {isDemand && summary.top_forecast?.length > 0 && (
        <div className="sum-chart">
          <p className="chart-title">상위 예측 품목</p>
          {summary.top_forecast.slice(0, 4).map((item, i) => {
            const max = summary.top_forecast[0].qty;
            const pct = Math.round((item.qty / max) * 100);
            return (
              <div key={i} className="bar-row">
                <span className="bar-label">{item.name}</span>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="bar-val">{item.qty}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="sum-footer">상세 분석 보기 →</div>
    </div>
  );
}

function StatItem({ label, value, color }) {
  return (
    <div className={`stat-item ${color ? 'stat-' + color : ''}`}>
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}
