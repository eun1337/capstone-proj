import { useEffect, useMemo, useState } from 'react';
import { Bar, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client.js';
import './KpiExplainerModal.css';

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;
const fmtQty = (n, unit) => `${Math.round(n).toLocaleString()}${unit ? ` ${unit}` : ''}`;
const fmtWonM = (n) => `₩${Math.round(n / 1_000_000)}M`;
const WEEKDAYS = ['일', '월', '화', '수', '목', '금', '토'];
const fmtShortDate = (iso) => {
  const d = new Date(`${iso}T00:00:00`);
  return `${d.getMonth() + 1}/${d.getDate()}`;
};

// 판매금액 KPI 모달 전용 — 일별 추이 콤보 차트 위 다크 알약 툴팁.
function TrendTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  const d = new Date(`${label}T00:00:00`);
  const dow = d.getDay();
  return (
    <div className="kpi-modal-trend-tooltip">
      <div className="kpi-modal-trend-tooltip-date">
        {fmtShortDate(label)} ({WEEKDAYS[dow]})
      </div>
      <div className="kpi-modal-trend-tooltip-row">
        <span className="kpi-modal-trend-dot solid" /> 판매금액 <strong>{fmtWon(row.sales_amount)}</strong>
      </div>
      <div className="kpi-modal-trend-tooltip-row">
        <span className="kpi-modal-trend-dot line" /> 7일 이동평균 <strong>{fmtWon(row.ma7)}</strong>
      </div>
    </div>
  );
}

// 판매건수 KPI 모달 전용 — 일별 판매건수 + 건당 평균 판매액(AOV) 콤보 차트 툴팁.
function CountTrendTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  const d = new Date(`${label}T00:00:00`);
  const dow = d.getDay();
  return (
    <div className="kpi-modal-trend-tooltip">
      <div className="kpi-modal-trend-tooltip-date">
        {fmtShortDate(label)} ({WEEKDAYS[dow]})
      </div>
      <div className="kpi-modal-trend-tooltip-row">
        <span className="kpi-modal-trend-dot indigo" /> 판매건수 <strong>{row.sales_record_count.toLocaleString()}건</strong>
      </div>
      <div className="kpi-modal-trend-tooltip-row">
        <span className="kpi-modal-trend-dot pink" /> 건당 평균금액 <strong>{fmtWon(row.aov)}</strong>
      </div>
    </div>
  );
}

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
  kpiKey, summary, center, operationalDate, categoryLabel, unit,
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

  // ── 판매금액 전용: 상단 일별 추이 콤보 차트 ─────────────────────────
  // summary.sparkline(최근 30일, 이미 부모가 받아온 값)을 그대로 데이터소스로 쓴다 —
  // 이 모달만을 위한 새 API 호출이 필요 없다(범위/카테고리 필터도 summary와 동일하게 이미 반영돼 있음).
  // 차트는 시계열 그래프라 항상 날짜순 그대로 두고, 정렬 토글은 아래 상품별 표에만 건다.
  const trendRows = useMemo(() => {
    const pts = summary?.sparkline;
    if (kpiKey !== 'sales' || !pts || pts.length === 0) return [];
    return pts.map((p, i) => {
      const windowPts = pts.slice(Math.max(0, i - 6), i + 1); // 7일 이동평균(초반엔 있는 만큼만 평균)
      const ma7 = windowPts.reduce((s, w) => s + w.sales_amount, 0) / windowPts.length;
      return { date: p.date, sales_amount: p.sales_amount, ma7 };
    });
  }, [kpiKey, summary]);

  // ── 판매건수 전용: 상단 일별 판매건수 + 건당 평균 판매액(AOV) 콤보 차트 ────────────
  // summary.sparkline은 sales_record_count/sales_amount를 이미 일별로 갖고 있어(최근 30일),
  // 그중 최근 7일만 잘라 쓴다 — AOV는 그날그날 값이라(이동평균이 아님) 굳이 30일씩 안 필요하다.
  const countTrendRows = useMemo(() => {
    const pts = summary?.sparkline;
    if (kpiKey !== 'count' || !pts || pts.length === 0) return [];
    return pts.slice(-7).map((p) => ({
      date: p.date,
      sales_record_count: p.sales_record_count,
      aov: p.sales_record_count > 0 ? p.sales_amount / p.sales_record_count : 0,
    }));
  }, [kpiKey, summary]);

  // 상품별 표 정렬 토글 — 'amount'(판매금액/판매건수 높은순, 기본값·기존 정렬과 동일) /
  // 'qty'(판매수량 높은순). 라벨 텍스트는 kpiKey별로 아래 JSX에서 다르게 붙인다.
  const [productSort, setProductSort] = useState('amount');

  function downloadProductCsv(rowsToExport) {
    let header, body, filename;
    if (kpiKey === 'count') {
      header = '순위,상품명,바코드,판매건수';
      body = rowsToExport
        .map((r, i) => `${i + 1},${r.product_name},${r.barcode},${r.record_count}`)
        .join('\n');
      filename = `판매건수_상품별내역_${center}_${operationalDate}.csv`;
    } else {
      header = '순위,상품명,바코드,판매수량,판매금액';
      body = rowsToExport
        .map((r, i) => `${i + 1},${r.product_name},${r.barcode},${Math.round(r.sales_qty)},${Math.round(r.sales_amount)}`)
        .join('\n');
      filename = `판매금액_상품별내역_${operationalDate}.csv`;
    }
    const blob = new Blob([`﻿${header}\n${body}`], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  const def = KPI_DEFINITIONS[kpiKey];
  if (!def || !summary) return null;

  let rows = [];
  if (items) {
    if (kpiKey === 'sales') {
      rows = items
        .filter((i) => i.sales_amount > 0)
        .map((i) => ({ ...i, net_sales_amount: i.sales_amount - i.return_amount }))
        .sort((a, b) => (productSort === 'qty' ? b.sales_qty - a.sales_qty : b.sales_amount - a.sales_amount));
    } else if (kpiKey === 'count') {
      rows = items
        .filter((i) => i.record_count > 0)
        .sort((a, b) => (
          productSort === 'name'
            ? a.product_name.localeCompare(b.product_name, 'ko')
            : b.record_count - a.record_count
        ));
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
  // 판매건수 컬럼 인라인 게이지 바 — 현재 표(검색 반영 전 전체 rows) 안에서 최댓값 대비 비율.
  const countMaxRecordCount = kpiKey === 'count' && rows.length ? Math.max(...rows.map((r) => r.record_count)) : 1;

  return (
    <div className="kpi-modal-backdrop" onClick={onClose}>
      <div className="kpi-modal" onClick={(e) => e.stopPropagation()}>
        <div className="kpi-modal-hd">
          <div>
            <h3>{def.title}</h3>
            <p>{center}센터{categoryLabel ? ` · ${categoryLabel}` : ''} · 조회일 {operationalDate}{unit ? ` · 단위 ${unit}` : ''}</p>
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

          {kpiKey === 'sales' && trendRows.length > 0 && (
            <div className="kpi-modal-trend-section">
              <div className="kpi-modal-trend-hd">
                <h4>일별 추이 (최근 {trendRows.length}일)</h4>
              </div>

              <div className="kpi-modal-trend-chart">
                <ResponsiveContainer width="100%" height={208}>
                  <ComposedChart data={trendRows} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
                    {/* 막대/이동평균선은 30일 전체를 그대로 쓰되(데이터를 좁히면 이동평균의 의미가
                        없어진다), x축 라벨은 기준일/기준일-7/-14/-21/-28, 이 5개 지점(1주 간격)만
                        렌더링한다 — interval={0}으로 모든 tick 자리는 유지하되 이 5개를 뺀 나머지는
                        tickFormatter가 빈 문자열을 반환해 자리만 차지하고 안 보인다. */}
                    <XAxis
                      dataKey="date" interval={0}
                      tickFormatter={(value, index) => {
                        const fromEnd = trendRows.length - 1 - index; // 0=기준일, 7=기준일-7, ...
                        return [0, 7, 14, 21, 28].includes(fromEnd) ? fmtShortDate(value) : '';
                      }}
                      tick={{ fontSize: 11, fill: '#64748b' }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false}
                    />
                    <YAxis
                      yAxisId="left" tickFormatter={fmtWonM}
                      tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} width={44}
                    />
                    <YAxis yAxisId="right" orientation="right" hide />
                    <Tooltip content={<TrendTooltip />} cursor={{ fill: 'rgba(148,163,184,0.1)' }} />
                    <Bar yAxisId="left" dataKey="sales_amount" fill="#3b82f6" radius={[3, 3, 0, 0]} barSize={8} />
                    <Line
                      yAxisId="right" dataKey="ma7" type="monotone"
                      stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 4 }}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {kpiKey === 'count' && countTrendRows.length > 0 && (
            <div className="kpi-modal-trend-section">
              <div className="kpi-modal-trend-hd">
                <h4>일별 추이 (최근 {countTrendRows.length}일)</h4>
              </div>

              <div className="kpi-modal-trend-chart">
                <ResponsiveContainer width="100%" height={208}>
                  <ComposedChart data={countTrendRows} margin={{ top: 6, right: 6, bottom: 0, left: 0 }}>
                    <XAxis
                      dataKey="date" interval={0} tickFormatter={fmtShortDate}
                      tick={{ fontSize: 11, fill: '#64748b' }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false}
                    />
                    <YAxis
                      yAxisId="left" tickFormatter={(v) => v.toLocaleString()}
                      tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} width={32}
                    />
                    <YAxis yAxisId="right" orientation="right" hide />
                    <Tooltip content={<CountTrendTooltip />} cursor={{ fill: 'rgba(148,163,184,0.1)' }} />
                    <Bar yAxisId="left" dataKey="sales_record_count" fill="#6366f1" radius={[3, 3, 0, 0]} barSize={16} />
                    <Line
                      yAxisId="right" dataKey="aov" type="monotone"
                      stroke="#ec4899" strokeWidth={2} dot={false} activeDot={{ r: 4 }}
                      isAnimationActive={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          <div className="kpi-modal-list-section">
            <div className="kpi-modal-list-hd">
              <span className="kpi-modal-list-title-group">
                <h4>
                  {kpiKey === 'sales' && '상품별 판매 상세 내역'}
                  {kpiKey === 'count' && '상품별 판매건수 상세 내역'}
                  {kpiKey === 'sku' && '상품 목록'}
                </h4>
                {!itemsLoading && !itemsError && (
                  <span className="kpi-modal-list-count">
                    {search.trim() ? `${shownRows.length.toLocaleString()} / ${totalRowCount.toLocaleString()}개` : `${totalRowCount.toLocaleString()}개`}
                  </span>
                )}
              </span>
              {(kpiKey === 'sales' || kpiKey === 'count') && (
                <div className="kpi-modal-list-controls">
                  <div className="kpi-modal-sort-toggle">
                    <button
                      type="button"
                      className={`kpi-modal-sort-btn ${productSort === 'amount' ? 'active' : ''}`}
                      onClick={() => setProductSort('amount')}
                    >
                      {kpiKey === 'sales' ? '금액 높은순' : '판매건수순'}
                    </button>
                    <button
                      type="button"
                      className={`kpi-modal-sort-btn ${productSort === (kpiKey === 'count' ? 'name' : 'qty') ? 'active' : ''}`}
                      onClick={() => setProductSort(kpiKey === 'count' ? 'name' : 'qty')}
                    >
                      {kpiKey === 'sales' ? '수량 높은순' : '상품명순'}
                    </button>
                  </div>
                  <button type="button" className="kpi-modal-csv-btn" onClick={() => downloadProductCsv(shownRows)}>
                    📥 CSV 다운로드
                  </button>
                </div>
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
                      <tr><th className="unit">순위</th><th className="left">상품명</th><th className="left">바코드</th><th>판매수량</th><th>판매금액</th></tr>
                    )}
                    {kpiKey === 'count' && (
                      <tr><th className="unit">순위</th><th className="left">상품명</th><th className="left">바코드</th><th>판매건수</th></tr>
                    )}
                    {kpiKey === 'sku' && (
                      <tr><th className="unit">단위</th><th className="left">상품</th><th className="left">바코드</th><th>판매수량</th><th>판매금액</th></tr>
                    )}
                  </thead>
                  <tbody>
                    {kpiKey === 'sales' && shownRows.map((r, i) => (
                      <tr key={r.sku_id}>
                        <td className="unit">{i + 1}</td>
                        <td className="left" title={r.product_name}><span className="kpi-modal-name">{r.product_name}</span></td>
                        <td className="left">{r.barcode}</td>
                        <td>{fmtQty(r.sales_qty, r.option_code)}</td>
                        <td className="kpi-modal-mono">{fmtWon(r.sales_amount)}</td>
                      </tr>
                    ))}
                    {kpiKey === 'count' && shownRows.map((r, i) => (
                      <tr key={r.sku_id}>
                        <td className="unit">{i + 1}</td>
                        <td className="left" title={r.product_name}><span className="kpi-modal-name">{r.product_name}</span></td>
                        <td className="left">{r.barcode}</td>
                        <td className="kpi-modal-gauge-cell">
                          <span
                            className="kpi-modal-gauge-fill kpi-modal-gauge-fill-indigo"
                            style={{ width: `${(r.record_count / countMaxRecordCount) * 100}%` }}
                          />
                          <span className="kpi-modal-gauge-text">{r.record_count.toLocaleString()}건</span>
                        </td>
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
