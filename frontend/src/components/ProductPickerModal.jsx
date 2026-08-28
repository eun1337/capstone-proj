import { useEffect, useMemo, useState } from 'react';
import { api } from '../api/client.js';
import './ProductPickerModal.css';

const UNIT_FILTERS = [
  { key: 'ALL', label: '전체' },
  { key: 'EA', label: 'EA' },
  { key: 'BX', label: 'BX' },
  { key: 'CS', label: 'CS' },
];
const FORECAST_FILTERS = [
  { key: 'all', label: '전체' },
  { key: 'available', label: '예측 가능' },
  { key: 'unavailable', label: '예측 없음' },
];
const SORT_OPTIONS = [
  { key: 'name', label: '이름순' },
  { key: 'recent_sales', label: '기준 판매수량 높은순' },
  { key: 'h1', label: '1주 예상수요 높은순' },
  { key: 'sales_amount', label: '판매금액 높은순' },
];

const fmtQty = (v) => (v === null || v === undefined ? '-' : Math.round(v).toLocaleString());
const fmtQty1 = (v) => (v === null || v === undefined ? '-' : v.toFixed(1));

// 상품 찾기 modal — Dashboard 메인과 분리된 탐색 기능. "전체" 단위 필터는 EA/BX/CS를
// 합산한다는 뜻이 아니라 모든 단위의 SKU를 검색 결과에 함께 보여준다는 뜻이다.
// 이 모달의 단위 필터는 상단 단위(topUnit)와 별개 — 선택 완료 시에만 상단 단위가
// 선택된 SKU의 실제 option_code로 바뀐다.
export default function ProductPickerModal({
  center, selectedPath, categoryLabel, operationalDate, aiBasisWeek, initialUnit,
  onCancel, onConfirm,
}) {
  const [large, middle, small] = selectedPath;

  const [rawProducts, setRawProducts] = useState([]);
  const [forecastMap, setForecastMap] = useState(new Map());
  const [activityMap, setActivityMap] = useState(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [search, setSearch] = useState('');
  const [unitFilter, setUnitFilter] = useState(initialUnit || 'ALL');
  const [forecastFilter, setForecastFilter] = useState('all');
  const [sortKey, setSortKey] = useState('name');
  const [pickedKey, setPickedKey] = useState(null);

  useEffect(() => {
    function handleKeyDown(e) {
      if (e.key === 'Escape') onCancel();
    }
    window.addEventListener('keydown', handleKeyDown);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = prevOverflow;
    };
  }, [onCancel]);

  useEffect(() => {
    let ignore = false;
    setLoading(true);
    setError(null);
    Promise.all([
      api.getDashboardProducts({ center, category_large: large, category_middle: middle, category_small: small }),
      api.getForecastProducts({ center, week_st: aiBasisWeek, category_large: large, category_middle: middle, category_small: small }),
      api.getDailyProductActivity({ center, date: operationalDate, category_large: large, category_middle: middle, category_small: small }),
    ])
      .then(([products, forecastProducts, activity]) => {
        if (ignore) return;
        setRawProducts(products);
        setForecastMap(new Map(forecastProducts.map((r) => [r.sku_id, r])));
        setActivityMap(new Map(activity.items.map((r) => [r.sku_id, r])));
      })
      .catch((e) => { if (!ignore) setError(e.message); })
      .finally(() => { if (!ignore) setLoading(false); });
    return () => { ignore = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [center, large, middle, small, operationalDate, aiBasisWeek]);

  const rows = useMemo(() => {
    return rawProducts.map((p) => {
      const fc = forecastMap.get(p.sku_id);
      const act = activityMap.get(p.sku_id);
      return {
        ...p,
        recent_sales_qty: fc ? fc.recent_sales_qty : null,
        h1_pred: fc ? fc.h1_pred : null,
        h2_pred: fc ? fc.h2_pred : null,
        h4_pred: fc ? fc.h4_pred : null,
        forecast_available_now: Boolean(fc),
        sales_amount: act ? act.sales_amount : 0,
      };
    });
  }, [rawProducts, forecastMap, activityMap]);

  const filteredSorted = useMemo(() => {
    const q = search.trim().toLowerCase();
    let list = rows.filter((p) => {
      if (q && !p.product_name.toLowerCase().includes(q) && !p.barcode.includes(q)) return false;
      if (unitFilter !== 'ALL' && p.option_code !== unitFilter) return false;
      if (forecastFilter === 'available' && !p.forecast_available_now) return false;
      if (forecastFilter === 'unavailable' && p.forecast_available_now) return false;
      return true;
    });
    list = [...list];
    if (sortKey === 'name') {
      list.sort((a, b) => a.product_name.localeCompare(b.product_name, 'ko'));
    } else if (sortKey === 'recent_sales') {
      list.sort((a, b) => (b.recent_sales_qty ?? -1) - (a.recent_sales_qty ?? -1));
    } else if (sortKey === 'h1') {
      list.sort((a, b) => (b.h1_pred ?? -1) - (a.h1_pred ?? -1));
    } else if (sortKey === 'sales_amount') {
      list.sort((a, b) => b.sales_amount - a.sales_amount);
    }
    return list;
  }, [rows, search, unitFilter, forecastFilter, sortKey]);

  const picked = filteredSorted.find((p) => p.sku_id === pickedKey) || null;

  return (
    <div className="ppm-backdrop" onClick={onCancel}>
      <div className="ppm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="ppm-hd">
          <div>
            <h3>{categoryLabel ? `${categoryLabel} 상품 보기` : '전체 상품 보기'}</h3>
            <p>선택 카테고리: {categoryLabel || '전체'}</p>
          </div>
          <button className="ppm-close" onClick={onCancel} aria-label="닫기">✕</button>
        </div>

        <div className="ppm-controls">
          <input
            type="text"
            className="ppm-search"
            placeholder="상품명 / 바코드 검색..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="ppm-filter-group">
            <span className="ppm-filter-label">단위</span>
            {UNIT_FILTERS.map((f) => (
              <button
                key={f.key}
                className={`filter-btn filter-btn-sm ${unitFilter === f.key ? 'active' : ''}`}
                onClick={() => setUnitFilter(f.key)}
              >{f.label}</button>
            ))}
          </div>
          <div className="ppm-filter-group">
            <span className="ppm-filter-label">예측 여부</span>
            {FORECAST_FILTERS.map((f) => (
              <button
                key={f.key}
                className={`filter-btn filter-btn-sm ${forecastFilter === f.key ? 'active' : ''}`}
                onClick={() => setForecastFilter(f.key)}
              >{f.label}</button>
            ))}
          </div>
          <select className="ppm-sort-select" value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
            {SORT_OPTIONS.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
          </select>
        </div>

        {loading && <div className="ppm-hint">불러오는 중...</div>}
        {!loading && error && <div className="ppm-hint ppm-hint-error">{error}</div>}

        {!loading && !error && (
          <div className="ppm-table-scroll">
            <table className="ppm-table">
              <thead>
                <tr>
                  <th className="ppm-col-radio"></th>
                  <th className="left">상품명</th>
                  <th className="left">바코드</th>
                  <th className="ppm-col-unit">단위</th>
                  <th>기준 판매수량</th>
                  <th>1주 예상수요</th>
                  <th>예측 여부</th>
                </tr>
              </thead>
              <tbody>
                {filteredSorted.map((p) => (
                  <tr
                    key={`${p.center_id}::${p.sku_id}`}
                    className={pickedKey === p.sku_id ? 'ppm-row-active' : ''}
                    onClick={() => setPickedKey(p.sku_id)}
                  >
                    <td className="ppm-col-radio">
                      <input type="radio" checked={pickedKey === p.sku_id} onChange={() => setPickedKey(p.sku_id)} />
                    </td>
                    <td className="left" title={p.product_name}>
                      <span className="ppm-name">{p.product_name}</span>
                    </td>
                    <td className="left">{p.barcode}</td>
                    <td className="ppm-col-unit">{p.option_code}</td>
                    <td>{fmtQty(p.recent_sales_qty)}</td>
                    <td>{fmtQty1(p.h1_pred)}</td>
                    <td>
                      <span className={`ppm-forecast-badge ${p.forecast_available_now ? 'ppm-forecast-yes' : 'ppm-forecast-no'}`}>
                        {p.forecast_available_now ? '예측 가능' : '예측 없음'}
                      </span>
                    </td>
                  </tr>
                ))}
                {filteredSorted.length === 0 && (
                  <tr><td colSpan={7} className="ppm-empty">조건에 맞는 상품이 없습니다.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        <div className="ppm-footer">
          <span className="ppm-total">총 {filteredSorted.length.toLocaleString()}개</span>
          <div className="ppm-footer-actions">
            <button className="ppm-btn ppm-btn-ghost" onClick={onCancel}>취소</button>
            <button
              className="ppm-btn ppm-btn-primary"
              disabled={!picked}
              onClick={() => picked && onConfirm(picked)}
            >선택 완료</button>
          </div>
        </div>
      </div>
    </div>
  );
}
