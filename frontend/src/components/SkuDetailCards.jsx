import './SkuDetailCards.css';

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;
const fmtNum = (n) => Math.round(n).toLocaleString();
const fmtNum1 = (n) => n.toFixed(1);

function InfoCard({ title, children }) {
  return (
    <div className="ins-card sdc-card">
      <div className="ins-card-hd"><h4>{title}</h4></div>
      <div className="sdc-body">{children}</div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="sdc-row">
      <span className="sdc-row-label">{label}</span>
      <span className="sdc-row-value">{value}</span>
    </div>
  );
}

function ChangeRow({ label, pct, absText }) {
  if (pct === null || pct === undefined) {
    return (
      <div className="sdc-row">
        <span className="sdc-row-label">{label}</span>
        <span className="sdc-row-value sdc-muted">-</span>
      </div>
    );
  }
  const isUp = pct >= 0;
  return (
    <div className="sdc-row">
      <span className="sdc-row-label">{label}</span>
      <span className="sdc-row-value">
        <span className={isUp ? 'sdc-up' : 'sdc-dn'}>{isUp ? '▲' : '▼'} {Math.abs(pct).toFixed(1)}%</span>
        {absText && <span className="sdc-abs"> {absText}</span>}
      </span>
    </div>
  );
}

// ── 1행 우측: 상품 정보 (SKU 선택 시 카테고리별 매출 TOP5 자리를 대체) ──────────
export function ProductInfoCard({ sku }) {
  if (!sku) return null;
  const kanPath = [sku.KAN_대분류, sku.KAN_중분류, sku.KAN_소분류].filter(Boolean).join(' > ');
  return (
    <InfoCard title="상품 정보">
      <Row label="상품명" value={<span className="sdc-strong" title={sku.product_name}>{sku.product_name}</span>} />
      <Row label="바코드" value={sku.barcode} />
      <Row label="단위" value={<span className="sdc-unit-badge">{sku.option_code}</span>} />
      <Row label="카테고리" value={<span className="sdc-kan" title={kanPath}>{kanPath || '-'}</span>} />
    </InfoCard>
  );
}

// ── 2행 우측: 현재 상태 (SKU 선택 시 판매지역별 매출 TOP5 자리를 대체) ──────────
// 추정재고/1주 예상수요/부족수량은 기존 /insights/inventory-shortage와 동일한 정의
// (max(h1_pred - 추정재고, 0))을 그대로 재사용한다 — 새 계산식이 아니다.
export function CurrentStatusCard({ unit, estimatedInventory, h1Pred, todayReturnQty, inventoryAvailable }) {
  const shortageQty = (estimatedInventory !== null && h1Pred !== null)
    ? Math.max(h1Pred - estimatedInventory, 0)
    : null;
  return (
    <InfoCard title="현재 상태">
      <Row label="추정재고" value={inventoryAvailable && estimatedInventory !== null ? `${fmtNum(estimatedInventory)} ${unit}` : '-'} />
      <Row label="1주 예상수요" value={h1Pred !== null ? `${fmtNum1(h1Pred)} ${unit}` : '예측 없음'} />
      <Row label="부족수량" value={shortageQty !== null ? `${fmtNum1(shortageQty)} ${unit}` : '-'} />
      <Row label="조회일 반품수량" value={`${fmtNum(todayReturnQty)} ${unit}`} />
    </InfoCard>
  );
}

// ── 3행 4개 카드 ──────────────────────────────────────────────
export function ForecastSummaryMiniCard({ unit, baseSalesQty, h1, h2, h4 }) {
  return (
    <InfoCard title="예측 요약">
      <Row label="기준 판매수량" value={baseSalesQty !== null ? `${fmtNum(baseSalesQty)} ${unit}` : '-'} />
      <Row label="1주 후" value={h1 !== null ? `${fmtNum1(h1)} ${unit}` : '예측 없음'} />
      <Row label="2주 후" value={h2 !== null ? `${fmtNum1(h2)} ${unit}` : '예측 없음'} />
      <Row label="4주 후" value={h4 !== null ? `${fmtNum1(h4)} ${unit}` : '예측 없음'} />
    </InfoCard>
  );
}

export function InventoryStatusMiniCard({ unit, estimatedInventory, h1Pred, inventoryAvailable }) {
  const shortageQty = (estimatedInventory !== null && h1Pred !== null)
    ? Math.max(h1Pred - estimatedInventory, 0)
    : null;
  return (
    <InfoCard title="재고 상태">
      <Row label="추정재고" value={inventoryAvailable && estimatedInventory !== null ? `${fmtNum(estimatedInventory)} ${unit}` : '-'} />
      <Row label="1주 예상수요" value={h1Pred !== null ? `${fmtNum1(h1Pred)} ${unit}` : '예측 없음'} />
      <Row label="부족수량" value={shortageQty !== null ? `${fmtNum1(shortageQty)} ${unit}` : '-'} />
    </InfoCard>
  );
}

export function SalesChangeMiniCard({ dayPct, dayAbsText, weekPct, weekAbsText }) {
  return (
    <InfoCard title="판매 변화">
      <ChangeRow label="전일 대비" pct={dayPct} absText={dayAbsText} />
      <ChangeRow label="전주 동일요일 대비" pct={weekPct} absText={weekAbsText} />
    </InfoCard>
  );
}

export function ReturnStatusMiniCard({ unit, returnQty, returnAmount, dayPct, dayAbsText }) {
  return (
    <InfoCard title="반품 현황">
      <Row label="조회일 반품수량" value={`${fmtNum(returnQty)} ${unit}`} />
      <Row label="조회일 반품금액" value={fmtWon(returnAmount)} />
      <ChangeRow label="전일 대비" pct={dayPct} absText={dayAbsText} />
    </InfoCard>
  );
}
