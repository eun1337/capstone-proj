import './MasterList.css';

const STATUS_MAP = {
  정상: { label: '정상', cls: 'status-ok' },
  주의: { label: '주의', cls: 'status-warn' },
  부족: { label: '부족', cls: 'status-bad' },
};

export default function MasterList({ products }) {
  if (!products.length) {
    return (
      <div className="ml-empty">
        <span>조건에 맞는 품목이 없습니다.</span>
      </div>
    );
  }

  return (
    <div className="ml-wrapper">
      <table className="ml-table">
        <thead>
          <tr>
            <th>바코드</th>
            <th>품목명</th>
            <th>카테고리</th>
            <th>센터</th>
            <th className="num-col">예측 수량</th>
            <th className="num-col">현재 재고</th>
            <th>재고 비율</th>
            <th>상태</th>
          </tr>
        </thead>
        <tbody>
          {products.map((p) => {
            const ratio  = p.forecast_qty > 0
              ? Math.min(Math.round((p.current_stock / p.forecast_qty) * 100), 200)
              : 0;
            const s      = STATUS_MAP[p.status] ?? { label: p.status, cls: '' };
            const barPct = Math.min(ratio, 100);
            const barCls = ratio >= 80 ? 'ratio-ok' : ratio >= 50 ? 'ratio-warn' : 'ratio-bad';

            return (
              <tr key={p.id}>
                <td className="barcode-col">{p.barcode}</td>
                <td className="name-col">{p.product_name}</td>
                <td>{p.category}</td>
                <td>
                  <span className={`center-tag ${p.center === 'A센터' ? 'center-a' : 'center-b'}`}>
                    {p.center}
                  </span>
                </td>
                <td className="num-col">{p.forecast_qty.toLocaleString()}</td>
                <td className="num-col">{p.current_stock.toLocaleString()}</td>
                <td className="ratio-col">
                  <div className="ratio-bar-track">
                    <div className={`ratio-bar-fill ${barCls}`} style={{ width: `${barPct}%` }} />
                  </div>
                  <span className="ratio-pct">{ratio}%</span>
                </td>
                <td>
                  <span className={`status-badge ${s.cls}`}>{s.label}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
