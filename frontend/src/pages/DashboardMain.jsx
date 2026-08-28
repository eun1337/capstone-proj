import { useState } from 'react';
import KpiCard from '../components/KpiCard.jsx';
import KpiExplainerModal from '../components/KpiExplainerModal.jsx';
import MainForecastChart from '../components/MainForecastChart.jsx';
import {
  CategoryOrProductTop5Card, CategoryOrProductDetail,
  RegionTop5Card, RegionDetail,
  AiTop5Card, AiTop5Detail,
  ShortageTop5Card, ShortageDetail,
  SurgeTop5Card, SurgeDetail,
  ReturnQtyTop5Card, ReturnQtyDetail,
  InsightModal,
} from '../components/TopRankCards.jsx';
import {
  ProductInfoCard, CurrentStatusCard,
  ForecastSummaryMiniCard, InventoryStatusMiniCard, SalesChangeMiniCard, ReturnStatusMiniCard,
} from '../components/SkuDetailCards.jsx';

const fmtWon = (n) => `₩${Math.round(n).toLocaleString()}`;
const fmtNum = (n) => Math.round(n).toLocaleString();

function computeChange(current, prev) {
  if (current === null || current === undefined || prev === null || prev === undefined || prev === 0) {
    return { pct: null, abs: null };
  }
  return { pct: ((current - prev) / prev) * 100, abs: current - prev };
}

function signed(n) {
  return n >= 0 ? `+${n}` : `${n}`;
}

const INSIGHT_TITLES = {
  category: '카테고리별 매출 TOP5',
  region: '판매지역별 매출 TOP5',
  ai: 'AI 예측 상품 TOP5',
  shortage: '1주 예상수요 대비 재고 부족',
  surge: '판매 증가 TOP5',
  returns: '오늘 반품수량 TOP5',
};

// 대시보드 메인 4-column grid — 카테고리/상품을 선택해도 grid 위치는 그대로 두고
// 같은 자리의 카드 내용만 갱신한다(요청사항 7). SKU 미선택(집계) 상태와 SKU 선택 상태
// 두 갈래로 나눠 같은 자리에 다른 컴포넌트를 채운다.
export default function DashboardMain({
  center, operationalDate, prevDate, prevWeekDate, formatDateLabel,
  topUnit, effectiveUnit,
  selectedPath, categoryLabel, selectedSku,
  aiBasisWeek, inventoryBasisWeek,
  summary, summaryLoading, summaryError, prevSummary,
  categorySales, categorySalesLoading, categorySalesError,
  regionSales, regionSalesLoading, regionSalesError,
  aiTop5, aiTop5Loading, aiTop5Error,
  shortage, shortageLoading, shortageError,
  surge, surgeLoading, surgeError,
  returnsRanking, returnsLoading, returnsError,
  demandTrend, demandTrendLoading, demandTrendError,
  forecast, forecastLoading, forecastError,
  inventory, inventoryLoading, inventoryError,
  transactions, transactionsLoading, transactionsError,
  onSelectCategoryFromRanking,
}) {
  const [large, middle, small] = selectedPath;
  const [openKpi, setOpenKpi] = useState(null);
  const [openRank, setOpenRank] = useState(null);

  const hasSku = Boolean(selectedSku);

  // ── KPI 값 계산 ──────────────────────────────────────────────
  let kpis = [];
  if (summary) {
    if (!hasSku) {
      const salesChange = computeChange(summary.total_sales_amount, prevSummary?.total_sales_amount);
      const countChange = computeChange(summary.sales_record_count, prevSummary?.sales_record_count);
      const skuChange = computeChange(summary.active_sku_count, prevSummary?.active_sku_count);
      kpis = [
        {
          key: 'sales', label: '판매금액', icon: '🔍', iconColor: '#3b82f6', color: '#3b82f6',
          value: fmtWon(summary.total_sales_amount),
          changePct: salesChange.pct,
          changeAbsText: salesChange.abs === null ? null : `${salesChange.abs >= 0 ? '+' : ''}${fmtWon(salesChange.abs)}`,
          sparkData: summary.sparkline.map((p) => p.sales_amount),
        },
        {
          key: 'count', label: '판매건수', icon: '🛒', iconColor: '#10b981',
          value: fmtNum(summary.sales_record_count),
          changePct: countChange.pct,
          changeAbsText: countChange.abs === null ? null : `${signed(Math.round(countChange.abs))}건`,
        },
        {
          key: 'sku', label: '판매 SKU 수', icon: '📦', iconColor: '#8b5cf6',
          value: fmtNum(summary.active_sku_count),
          changePct: skuChange.pct,
          changeAbsText: skuChange.abs === null ? null : `${signed(Math.round(skuChange.abs))}개`,
        },
      ];
    } else {
      const skuSalesQty = Object.values(summary.sales_qty_by_unit || {})[0] ?? 0;
      const prevSkuSalesQty = prevSummary ? (Object.values(prevSummary.sales_qty_by_unit || {})[0] ?? 0) : null;
      const salesChange = computeChange(summary.total_sales_amount, prevSummary?.total_sales_amount);
      const qtyChange = computeChange(skuSalesQty, prevSkuSalesQty);
      const countChange = computeChange(summary.sales_record_count, prevSummary?.sales_record_count);
      kpis = [
        {
          key: 'sales', label: '판매금액', icon: '🔍', iconColor: '#3b82f6', color: '#3b82f6',
          value: fmtWon(summary.total_sales_amount),
          changePct: salesChange.pct,
          changeAbsText: salesChange.abs === null ? null : `${salesChange.abs >= 0 ? '+' : ''}${fmtWon(salesChange.abs)}`,
          sparkData: summary.sparkline.map((p) => p.sales_amount),
        },
        {
          key: 'qty', label: '판매수량', icon: '🛒', iconColor: '#10b981',
          value: `${fmtNum(skuSalesQty)} ${effectiveUnit}`,
          changePct: qtyChange.pct,
          changeAbsText: qtyChange.abs === null ? null : `${signed(Math.round(qtyChange.abs))} ${effectiveUnit}`,
        },
        {
          key: 'count', label: '판매기록 수', icon: '📦', iconColor: '#8b5cf6',
          value: fmtNum(summary.sales_record_count),
          changePct: countChange.pct,
          changeAbsText: countChange.abs === null ? null : `${signed(Math.round(countChange.abs))}건`,
        },
      ];
    }
  }

  // ── SKU 선택 모드 전용 파생값 ────────────────────────────────
  const h1Pred = forecast?.h1?.predicted_qty ?? null;
  const h2Pred = forecast?.h2?.predicted_qty ?? null;
  const h4Pred = forecast?.h4?.predicted_qty ?? null;
  const baseSalesQty = forecast?.history?.length ? forecast.history[forecast.history.length - 1].sales_qty : null;
  const estimatedInventory = inventory?.inventory_available ? inventory.estimated_inventory : null;

  const txHistory = transactions?.history || [];
  const findByDate = (d) => txHistory.find((h) => h.date === d) || null;
  const todayTx = findByDate(operationalDate);
  const prevDayTx = findByDate(prevDate);
  const prevWeekTx = findByDate(prevWeekDate);
  const todayReturnQty = todayTx ? todayTx.return_qty : 0;
  const todayReturnAmount = todayTx ? todayTx.return_amount : 0;

  const salesDayChange = (todayTx && prevDayTx) ? computeChange(todayTx.sales_qty, prevDayTx.sales_qty) : { pct: null, abs: null };
  const salesWeekChange = (todayTx && prevWeekTx) ? computeChange(todayTx.sales_qty, prevWeekTx.sales_qty) : { pct: null, abs: null };
  const returnDayChange = (todayTx && prevDayTx) ? computeChange(todayTx.return_qty, prevDayTx.return_qty) : { pct: null, abs: null };

  const gridStyle = { display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '12px' };

  return (
    <>
      <div className="dm-page-hd">
        <h2>{categoryLabel || (selectedSku ? selectedSku.product_name : '전체')}</h2>
        <p>
          {center}센터 {categoryLabel ? `· ${categoryLabel} ` : ''}
          {selectedSku ? `· ${selectedSku.product_name} · ${selectedSku.option_code} ` : ''}
          기준 데이터입니다. 카테고리 또는 상품을 선택하여 더 자세한 정보를 확인할 수 있습니다.
        </p>
      </div>

      <div style={gridStyle}>
        {/* ── Row 1 ── */}
        {summaryLoading && <div className="dm-row-hint">KPI 불러오는 중...</div>}
        {!summaryLoading && summaryError && <div className="dm-row-hint dm-row-hint-error">{summaryError}</div>}
        {!summaryLoading && !summaryError && kpis.map((kpi) => (
          <KpiCard key={kpi.key} {...kpi} onClick={() => setOpenKpi(kpi.key)} />
        ))}

        {!hasSku ? (
          <CategoryOrProductTop5Card
            data={categorySales} loading={categorySalesLoading} error={categorySalesError}
            onOpenDetail={() => setOpenRank('category')}
          />
        ) : (
          <ProductInfoCard sku={selectedSku} />
        )}

        {/* ── Row 2: 차트(3칸) + 우측 1칸 ── */}
        <div className="chart-card dm-chart-card" style={{ gridColumn: 'span 3' }}>
          <div className="chart-card-hd">
            <div>
              <h3>수요예측 추이</h3>
              <p>
                {categoryLabel ? `${categoryLabel} · ` : ''}{center}센터 · 단위 {effectiveUnit}
                {aiBasisWeek ? ` · 기준주 ${aiBasisWeek}` : ''}
              </p>
            </div>
          </div>
          <MainForecastChart
            trend={hasSku ? (forecast && { history: forecast.history, h1: forecast.h1, h2: forecast.h2, h4: forecast.h4 }) : demandTrend}
            loading={hasSku ? forecastLoading : demandTrendLoading}
            error={hasSku ? forecastError : demandTrendError}
            unit={effectiveUnit}
          />
        </div>

        {!hasSku ? (
          <RegionTop5Card
            data={regionSales} loading={regionSalesLoading} error={regionSalesError}
            onOpenDetail={() => setOpenRank('region')}
          />
        ) : (
          <CurrentStatusCard
            unit={effectiveUnit}
            estimatedInventory={estimatedInventory}
            inventoryAvailable={Boolean(inventory?.inventory_available)}
            h1Pred={h1Pred}
            todayReturnQty={todayReturnQty}
          />
        )}

        {/* ── Row 3: 4칸 ── */}
        {!hasSku ? (
          <>
            <AiTop5Card data={aiTop5} loading={aiTop5Loading} error={aiTop5Error} onOpenDetail={() => setOpenRank('ai')} />
            <ShortageTop5Card data={shortage} loading={shortageLoading} error={shortageError} onOpenDetail={() => setOpenRank('shortage')} />
            <SurgeTop5Card data={surge} loading={surgeLoading} error={surgeError} onOpenDetail={() => setOpenRank('surge')} />
            <ReturnQtyTop5Card data={returnsRanking} loading={returnsLoading} error={returnsError} onOpenDetail={() => setOpenRank('returns')} />
          </>
        ) : (
          <>
            <ForecastSummaryMiniCard unit={effectiveUnit} baseSalesQty={baseSalesQty} h1={h1Pred} h2={h2Pred} h4={h4Pred} />
            <InventoryStatusMiniCard
              unit={effectiveUnit} estimatedInventory={estimatedInventory}
              inventoryAvailable={Boolean(inventory?.inventory_available)} h1Pred={h1Pred}
            />
            <SalesChangeMiniCard
              dayPct={salesDayChange.pct} dayAbsText={salesDayChange.abs === null ? null : `${signed(Math.round(salesDayChange.abs))} ${effectiveUnit}`}
              weekPct={salesWeekChange.pct} weekAbsText={salesWeekChange.abs === null ? null : `${signed(Math.round(salesWeekChange.abs))} ${effectiveUnit}`}
            />
            <ReturnStatusMiniCard
              unit={effectiveUnit} returnQty={todayReturnQty} returnAmount={todayReturnAmount}
              dayPct={returnDayChange.pct} dayAbsText={returnDayChange.abs === null ? null : `${signed(Math.round(returnDayChange.abs))} ${effectiveUnit}`}
            />
          </>
        )}
      </div>

      {openKpi && summary && (
        <KpiExplainerModal
          kpiKey={openKpi} summary={summary}
          center={center} operationalDate={operationalDate} categoryLabel={categoryLabel}
          categoryLarge={large} categoryMiddle={middle} categorySmall={small}
          onClose={() => setOpenKpi(null)}
        />
      )}

      {openRank && (
        <InsightModal
          title={INSIGHT_TITLES[openRank]}
          center={center}
          dateLabel={(openRank === 'ai' || openRank === 'shortage') ? '기준주' : '조회일'}
          basisWeek={(openRank === 'ai' || openRank === 'shortage') ? aiBasisWeek : operationalDate}
          wide={openRank === 'ai'}
          onClose={() => setOpenRank(null)}
        >
          {openRank === 'category' && categorySales && (
            <CategoryOrProductDetail data={categorySales} onSelectCategory={(label) => { onSelectCategoryFromRanking(label); setOpenRank(null); }} />
          )}
          {openRank === 'region' && regionSales && <RegionDetail data={regionSales} />}
          {openRank === 'ai' && <AiTop5Detail data={aiTop5} />}
          {openRank === 'shortage' && shortage && <ShortageDetail data={shortage} />}
          {openRank === 'surge' && surge && <SurgeDetail data={surge} />}
          {openRank === 'returns' && <ReturnQtyDetail data={returnsRanking} />}
        </InsightModal>
      )}
    </>
  );
}
