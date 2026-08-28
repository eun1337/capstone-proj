import { useState } from 'react';
import InsightCard from './InsightCard.jsx';
import InsightModal from './InsightModal.jsx';
import './TopRankCards.css';

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;
const fmtNum = (n) => Math.round(n).toLocaleString();
const fmtNum1 = (n) => n.toFixed(1);
const fmtPct = (v) => (v === null || v === undefined ? '-' : `${v.toFixed(1)}%`);

// 대시보드 하단(우측 2개 + 아래 4개) TOP5 카드 6종 + 각 "전체보기" modal.
// 상단에서 이미 단위(EA/BX/CS)가 확정돼 있으므로, 여기서는 행마다 단위 배지/단위 접미사를
// 반복하지 않는다(단위 표시는 modal 표에서만 별도 컬럼으로). 비율(비중/충족률/증가율)은
// 요청대로 메인 카드에서는 빼고 modal 표에만 남긴다.

/* ══════════════════════════════════════════════════════════════
   1. 카테고리별 매출 TOP5 / 상품별 매출 TOP5 (같은 슬롯, 카테고리 선택 시 전환)
   ══════════════════════════════════════════════════════════════ */
export function CategoryOrProductTop5Card({ data, loading, error, onOpenDetail }) {
  const isProductView = Boolean(data?.product_ranking);
  const items = isProductView ? data?.product_top5 : data?.top5;
  const isEmpty = !data || !items || items.length === 0 || data.total_sales_amount === 0;
  const title = isProductView ? '상품별 매출 TOP5' : '카테고리별 매출 TOP5';

  return (
    <InsightCard
      title={title}
      loading={loading}
      error={error}
      isEmpty={isEmpty}
      emptyMessage="해당 조회일에 판매 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items?.map((row) => (
          <div key={isProductView ? row.sku_id : row.category} className="trc-row">
            <span className="trc-name" title={isProductView ? row.product_name : row.category}>
              {isProductView ? row.product_name : row.category}
            </span>
            <span className="trc-val">{fmtWon(row.sales_amount)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function CategoryOrProductDetail({ data, onSelectCategory }) {
  const isProductView = Boolean(data.product_ranking);
  const [search, setSearch] = useState('');
  const allRows = isProductView ? data.product_ranking : data.ranking;
  const q = search.trim().toLowerCase();
  const rows = (isProductView && q)
    ? allRows.filter((row) => row.product_name.toLowerCase().includes(q) || row.barcode.includes(q))
    : allRows;
  return (
    <>
      {isProductView && (
        <input
          type="text"
          className="trc-search"
          placeholder="상품명 / 바코드 검색..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      )}
      <table className="trc-detail-table">
        <thead>
          {isProductView ? (
            <tr><th className="left">상품명</th><th>단위</th><th>판매금액</th></tr>
          ) : (
            <tr><th className="left">카테고리</th><th>판매금액</th></tr>
          )}
        </thead>
        <tbody>
          {rows.map((row) => (
            isProductView ? (
              <tr key={row.sku_id}>
                <td className="left">{row.product_name}</td>
                <td>{row.option_code}</td>
                <td className="num">{fmtWon(row.sales_amount)}</td>
              </tr>
            ) : (
              <tr
                key={row.category}
                className="trc-clickable-row"
                onClick={() => onSelectCategory(row.category)}
              >
                <td className="left">{row.category}</td>
                <td className="num">{fmtWon(row.sales_amount)}</td>
              </tr>
            )
          ))}
          {isProductView && rows.length === 0 && (
            <tr><td colSpan={3} className="trc-empty-row">조건에 맞는 상품이 없습니다.</td></tr>
          )}
        </tbody>
      </table>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════
   2. 판매지역별 매출 TOP5
   ══════════════════════════════════════════════════════════════ */
export function RegionTop5Card({ data, loading, error, onOpenDetail }) {
  return (
    <InsightCard
      title="판매지역별 매출 TOP5"
      loading={loading}
      error={error}
      isEmpty={!data || data.top5.length === 0}
      emptyMessage="해당 조회일에 판매 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {data?.top5.map((r) => (
          <div key={`${r.sido}-${r.sigungu}`} className="trc-row">
            <span className="trc-name-group" title={`${r.sigungu} · ${r.sido}`}>
              <span className="trc-name">{r.sigungu}</span>
              <span className="trc-sub">{r.sido}</span>
            </span>
            <span className="trc-val">{fmtWon(r.sales_amount)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function RegionDetail({ data }) {
  return (
    <table className="trc-detail-table">
      <thead>
        <tr><th className="left">시도</th><th className="left">시군구</th><th>판매금액</th></tr>
      </thead>
      <tbody>
        {data.ranking.map((r) => (
          <tr key={`${r.sido}-${r.sigungu}`}>
            <td className="left">{r.sido}</td>
            <td className="left">{r.sigungu}</td>
            <td className="num">{fmtWon(r.sales_amount)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ══════════════════════════════════════════════════════════════
   3. AI 예측 상품 TOP5
   ══════════════════════════════════════════════════════════════ */
export function AiTop5Card({ data, loading, error, onOpenDetail }) {
  const items = (data || []).slice(0, 5);
  return (
    <InsightCard
      title="AI 예측 상품 TOP5"
      loading={loading}
      error={error}
      isEmpty={items.length === 0}
      emptyMessage="해당 기준주에 예측 가능한 상품이 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.map((p) => (
          <div key={p.sku_id} className="trc-row">
            <span className="trc-name" title={p.product_name}>{p.product_name}</span>
            <span className="trc-val">{fmtNum1(p.h1_pred)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function AiTop5Detail({ data }) {
  return (
    <table className="trc-detail-table trc-detail-table-wide">
      <thead>
        <tr><th className="left">상품명</th><th className="left">바코드</th><th>단위</th><th>기준 판매수량</th><th>1주 예상수요</th><th>2주 예상수요</th><th>4주 예상수요</th></tr>
      </thead>
      <tbody>
        {data.map((p) => (
          <tr key={p.sku_id}>
            <td className="left">{p.product_name}</td>
            <td className="left">{p.barcode}</td>
            <td>{p.option_code}</td>
            <td className="num">{p.recent_sales_qty === null ? '-' : fmtNum(p.recent_sales_qty)}</td>
            <td className="num">{p.h1_pred === null ? '-' : fmtNum1(p.h1_pred)}</td>
            <td className="num">{p.h2_pred === null ? '-' : fmtNum1(p.h2_pred)}</td>
            <td className="num">{p.h4_pred === null ? '-' : fmtNum1(p.h4_pred)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ══════════════════════════════════════════════════════════════
   4. 1주 예상수요 대비 재고 부족 TOP5
   ══════════════════════════════════════════════════════════════ */
export function ShortageTop5Card({ data, loading, error, onOpenDetail }) {
  const items = data?.top5 || [];
  return (
    <InsightCard
      title="1주 예상수요 대비 재고 부족 TOP5"
      loading={loading}
      error={error}
      isEmpty={!data || items.length === 0}
      emptyMessage="해당 기준주에 비교 가능한 재고/예측 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.map((s) => (
          <div key={s.sku_id} className="trc-row">
            <span className="trc-name" title={s.product_name}>{s.product_name}</span>
            <span className="trc-val">{fmtNum(s.shortage_qty)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function ShortageDetail({ data }) {
  return (
    <table className="trc-detail-table">
      <thead>
        <tr><th className="left">상품명</th><th className="left">바코드</th><th>단위</th><th>추정재고</th><th>1주 예상수요</th><th>부족수량</th></tr>
      </thead>
      <tbody>
        {data.ranking.map((s) => (
          <tr key={s.sku_id}>
            <td className="left">{s.product_name}</td>
            <td className="left">{s.barcode}</td>
            <td>{s.option_code}</td>
            <td className="num">{fmtNum(s.estimated_inventory)}</td>
            <td className="num">{fmtNum1(s.h1_pred)}</td>
            <td className="num">{fmtNum1(s.shortage_qty)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ══════════════════════════════════════════════════════════════
   5. 판매 증가 TOP5
   ══════════════════════════════════════════════════════════════ */
export function SurgeTop5Card({ data, loading, error, onOpenDetail }) {
  const items = data?.top5 || [];
  return (
    <InsightCard
      title="판매 증가 TOP5"
      loading={loading}
      error={error}
      isEmpty={!data || items.length === 0}
      emptyMessage="전주 동일요일 대비 판매가 증가한 상품이 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.map((s) => (
          <div key={s.sku_id} className="trc-row">
            <span className="trc-name" title={s.product_name}>{s.product_name}</span>
            <span className="trc-val">+{fmtNum(s.increase_qty)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function SurgeDetail({ data }) {
  return (
    <table className="trc-detail-table">
      <thead>
        <tr><th className="left">상품명</th><th className="left">바코드</th><th>단위</th><th>비교 기준 판매수량</th><th>현재 판매수량</th><th>증가수량</th><th>증가율</th></tr>
      </thead>
      <tbody>
        {data.ranking.map((s) => (
          <tr key={s.sku_id}>
            <td className="left">{s.product_name}</td>
            <td className="left">{s.barcode}</td>
            <td>{s.option_code}</td>
            <td className="num">{fmtNum(s.prev_sales_qty)}</td>
            <td className="num">{fmtNum(s.current_sales_qty)}</td>
            <td className="num">+{fmtNum(s.increase_qty)}</td>
            <td className="num">{s.increase_pct === null ? '-' : `${s.increase_pct.toFixed(1)}%`}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ══════════════════════════════════════════════════════════════
   6. 오늘 반품수량 TOP5 — daily/returns(ranking)를 반품수량 기준으로 재정렬해 쓴다
      (금액 정렬 endpoint를 그대로 재사용, 새 backend 없이 클라이언트에서만 정렬 변경).
   ══════════════════════════════════════════════════════════════ */
export function ReturnQtyTop5Card({ data, loading, error, onOpenDetail }) {
  const items = data || [];
  return (
    <InsightCard
      title="오늘 반품수량 TOP5"
      loading={loading}
      error={error}
      isEmpty={items.length === 0}
      emptyMessage="해당 조회일에 반품 데이터가 없습니다."
      onOpenDetail={onOpenDetail}
    >
      <div className="ins-ranking-list">
        {items.slice(0, 5).map((r) => (
          <div key={r.sku_id} className="trc-row">
            <span className="trc-name" title={r.product_name}>{r.product_name}</span>
            <span className="trc-val">{fmtNum(r.return_qty)}</span>
          </div>
        ))}
      </div>
    </InsightCard>
  );
}

export function ReturnQtyDetail({ data }) {
  return (
    <table className="trc-detail-table">
      <thead>
        <tr><th className="left">상품명</th><th className="left">바코드</th><th>단위</th><th>반품수량</th><th>반품금액</th></tr>
      </thead>
      <tbody>
        {data.map((r) => (
          <tr key={r.sku_id}>
            <td className="left">{r.product_name}</td>
            <td className="left">{r.barcode}</td>
            <td>{r.option_code}</td>
            <td className="num">{fmtNum(r.return_qty)}</td>
            <td className="num">{fmtWon(r.return_amount)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export { InsightModal };
