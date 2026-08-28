import { useEffect, useState } from 'react';
import { api } from '../api/client.js';
import './KpiExplainerModal.css';

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;
const fmtQty = (n, unit) => `${Math.round(n).toLocaleString()}${unit ? ` ${unit}` : ''}`;

// 대시보드 KPI 3개(판매금액/판매건수/판매 SKU 수)가 공유하는 설명 modal.
// 정의는 새 계산 없이 고정 텍스트, 상품 목록은 GET /daily/product-activity 하나를
// 받아 KPI별로 필터링/정렬만 여기서 한다(상품별 새 반품률 등은 계산하지 않는다).
const KPI_DEFINITIONS = {
  sales: {
    title: '판매금액',
    description: '조회일에 발생한 판매 거래의 판매금액 합계입니다. 반품금액은 차감하지 않습니다.',
  },
  count: {
    title: '판매건수',
    description: '조회일에 발생한 원본 매출 기록(거래 행) 수입니다. 주문번호가 없는 데이터라 주문 단위가 아닌 거래 기록 단위입니다.',
  },
  sku: {
    title: '판매 SKU 수',
    description: '조회일에 판매수량이 1개 이상 발생한 서로 다른 SKU 수입니다.',
  },
};

export default function KpiExplainerModal({
  kpiKey, summary, center, operationalDate, categoryLabel,
  categoryLarge, categoryMiddle, categorySmall,
  onClose,
}) {
  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', handleKeyDown);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = prevOverflow;
    };
  }, [onClose]);

  const [items, setItems] = useState(null);
  const [itemsLoading, setItemsLoading] = useState(true);
  const [itemsError, setItemsError] = useState(null);
  const [search, setSearch] = useState('');

  useEffect(() => {
    let ignore = false;
    setItemsLoading(true);
    setItemsError(null);
    api.getDailyProductActivity({
      center, date: operationalDate,
      category_large: categoryLarge, category_middle: categoryMiddle, category_small: categorySmall,
    })
      .then((data) => { if (!ignore) setItems(data.items); })
      .catch((e) => { if (!ignore) { setItemsError(e.message); setItems(null); } })
      .finally(() => { if (!ignore) setItemsLoading(false); });
    return () => { ignore = true; };
  }, [center, operationalDate, categoryLarge, categoryMiddle, categorySmall]);

  const def = KPI_DEFINITIONS[kpiKey];
  if (!def || !summary) return null;

  let rows = [];
  if (items) {
    if (kpiKey === 'sales') {
      rows = items
        .filter((i) => i.sales_amount > 0)
        .map((i) => ({ ...i, net_sales_amount: i.sales_amount - i.return_amount }))
        .sort((a, b) => b.sales_amount - a.sales_amount);
    } else if (kpiKey === 'count') {
      rows = items
        .filter((i) => i.record_count > 0)
        .sort((a, b) => b.record_count - a.record_count);
    } else if (kpiKey === 'sku') {
      rows = items
        .filter((i) => i.sales_qty > 0)
        .sort((a, b) => b.sales_amount - a.sales_amount);
    }
  }
  const totalRowCount = rows.length;
  const q = search.trim().toLowerCase();
  const shownRows = q
    ? rows.filter((r) => r.product_name.toLowerCase().includes(q) || r.barcode.includes(q))
    : rows;

  return (
    <div className="kpi-modal-backdrop" onClick={onClose}>
      <div className="kpi-modal" onClick={(e) => e.stopPropagation()}>
        <div className="kpi-modal-hd">
          <div>
            <h3>{def.title}</h3>
            <p>{center}센터{categoryLabel ? ` · ${categoryLabel}` : ''} · 조회일 {operationalDate}</p>
          </div>
          <button className="kpi-modal-close" onClick={onClose} aria-label="닫기">✕</button>
        </div>

        <div className="kpi-modal-body">
          <p className="kpi-modal-desc">{def.description}</p>
          {kpiKey === 'sales' && <p className="kpi-modal-formula">순판매금액 = 판매금액 - 반품금액</p>}

          <div className="kpi-modal-values">
            {kpiKey === 'sales' && (
              <>
                <div className="kpi-modal-value-row kpi-modal-value-row-primary">
                  <span>판매금액</span><strong>{fmtWon(summary.total_sales_amount)}</strong>
                </div>
                <div className="kpi-modal-value-row">
                  <span>반품금액</span><strong>{fmtWon(summary.return_amount)}</strong>
                </div>
                <div className="kpi-modal-value-row">
                  <span>순판매금액</span><strong>{fmtWon(summary.net_sales_amount)}</strong>
                </div>
              </>
            )}
            {kpiKey === 'count' && (
              <div className="kpi-modal-value-row kpi-modal-value-row-primary">
                <span>판매건수</span><strong>{summary.sales_record_count.toLocaleString()}건</strong>
              </div>
            )}
            {kpiKey === 'sku' && (
              <div className="kpi-modal-value-row kpi-modal-value-row-primary">
                <span>판매 SKU 수</span><strong>{summary.active_sku_count.toLocaleString()}</strong>
              </div>
            )}
          </div>

          <div className="kpi-modal-list-section">
            <div className="kpi-modal-list-hd">
              <h4>상품 목록</h4>
              {!itemsLoading && !itemsError && (
                <span className="kpi-modal-list-count">
                  {search.trim() ? `${shownRows.length.toLocaleString()} / ${totalRowCount.toLocaleString()}개` : `${totalRowCount.toLocaleString()}개`}
                </span>
              )}
            </div>

            <input
              type="text"
              className="kpi-modal-search"
              placeholder="상품명 / 바코드 검색..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />

            {itemsLoading && <div className="kpi-modal-list-hint">불러오는 중...</div>}
            {!itemsLoading && itemsError && <div className="kpi-modal-list-hint">{itemsError}</div>}
            {!itemsLoading && !itemsError && shownRows.length === 0 && (
              <div className="kpi-modal-list-hint">해당 조건의 상품이 없습니다.</div>
            )}

            {!itemsLoading && !itemsError && shownRows.length > 0 && (
              <div className="kpi-modal-table-scroll">
                <table className="kpi-modal-table">
                  <thead>
                    {kpiKey === 'sales' && (
                      <tr><th className="unit">단위</th><th className="left">상품</th><th className="left">바코드</th><th>판매수량</th><th>판매금액</th><th>반품금액</th><th>순판매금액</th></tr>
                    )}
                    {kpiKey === 'count' && (
                      <tr><th className="unit">단위</th><th className="left">상품</th><th className="left">바코드</th><th>판매기록 수</th><th>판매수량</th><th>판매금액</th></tr>
                    )}
                    {kpiKey === 'sku' && (
                      <tr><th className="unit">단위</th><th className="left">상품</th><th className="left">바코드</th><th>판매수량</th><th>판매금액</th></tr>
                    )}
                  </thead>
                  <tbody>
                    {kpiKey === 'sales' && shownRows.map((r) => (
                      <tr key={r.sku_id}>
                        <td className="unit"><span className="kpi-modal-unit-badge">{r.option_code}</span></td>
                        <td className="left" title={r.product_name}><span className="kpi-modal-name">{r.product_name}</span></td>
                        <td className="left">{r.barcode}</td>
                        <td>{fmtQty(r.sales_qty)}</td>
                        <td>{fmtWon(r.sales_amount)}</td>
                        <td>{fmtWon(r.return_amount)}</td>
                        <td>{fmtWon(r.net_sales_amount)}</td>
                      </tr>
                    ))}
                    {kpiKey === 'count' && shownRows.map((r) => (
                      <tr key={r.sku_id}>
                        <td className="unit"><span className="kpi-modal-unit-badge">{r.option_code}</span></td>
                        <td className="left" title={r.product_name}><span className="kpi-modal-name">{r.product_name}</span></td>
                        <td className="left">{r.barcode}</td>
                        <td>{r.record_count.toLocaleString()}</td>
                        <td>{fmtQty(r.sales_qty)}</td>
                        <td>{fmtWon(r.sales_amount)}</td>
                      </tr>
                    ))}
                    {kpiKey === 'sku' && shownRows.map((r) => (
                      <tr key={r.sku_id}>
                        <td className="unit"><span className="kpi-modal-unit-badge">{r.option_code}</span></td>
                        <td className="left" title={r.product_name}><span className="kpi-modal-name">{r.product_name}</span></td>
                        <td className="left">{r.barcode}</td>
                        <td>{fmtQty(r.sales_qty)}</td>
                        <td>{fmtWon(r.sales_amount)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
