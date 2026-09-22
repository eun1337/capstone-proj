import { useState } from 'react';
import KpiCard from '../components/KpiCard.jsx';
import KpiExplainerModal from '../components/KpiExplainerModal.jsx';
import MainForecastChart from '../components/MainForecastChart.jsx';
import HistoryRangeControl from '../components/HistoryRangeControl.jsx';
import {
  CategoryOrProductTop5Card, CategoryOrProductDetail, categoryDrilldownInfo,
  RegionTop5Card, RegionDetail, RegionViewToggle,
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

// sparkline 배열(DailySparklinePoint[], 항상 최근 30일 dense) → KpiCard용
// { v, isRecent, date }[]로 변환하면서 기간 토글(rangeDays: 7 | 30)만큼 뒤에서 잘라낸다.
// Tableau 스펙: 마지막 7개가 최근구간(isRecent=true) — 7일 뷰에서는 전부 최근구간이 된다.
function buildSparkData(sparkline, key, rangeDays = 30) {
  if (!sparkline || sparkline.length === 0) return undefined;
  const sliced = rangeDays ? sparkline.slice(-rangeDays) : sparkline;
  const recentThreshold = sliced.length - 7;
  return sliced.map((p, i) => ({ v: p[key] ?? 0, isRecent: i >= recentThreshold, date: p.date }));
}

function computeChange(current, prev) {
  if (current === null || current === undefined || prev === null || prev === undefined || prev === 0) {
    return { pct: null, abs: null };
  }
  return { pct: ((current - prev) / prev) * 100, abs: current - prev };
}

// 전주 대비(WoW) — 기준일(sparkline 마지막 값) vs 그 7일 전(같은 요일) 값의 단순 비교.
// sparkline은 기간 토글과 무관하게 항상 30일 전체가 내려오므로, 이 계산은 토글 상태와
// 상관없이 언제나 "기준일 vs 기준일-7일" 한 쌍만 본다.
function computeWowChange(sparkline, key) {
  if (!sparkline || sparkline.length < 8) return { pct: null };
  const n = sparkline.length;
  return computeChange(sparkline[n - 1]?.[key], sparkline[n - 8]?.[key]);
}

function signed(n) {
  return n >= 0 ? `+${n}` : `${n}`;
}

// category는 드릴다운 깊이에 따라 제목이 바뀌므로 categoryDrilldownInfo()로 동적으로 계산한다
// (아래 목록엔 넣지 않음).
const INSIGHT_TITLES = {
  region: '권역별 매출 현황',
  ai: '예측 출고량 상위 TOP 5',
  shortage: '결품 위험 상품 TOP 5',
  surge: '판매 급증 상품 TOP 5',
  returns: '반품 발생 상위 TOP 5',
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
  historyWeeks, onHistoryWeeksChange,
}) {
  const [large, middle, small] = selectedPath;
  const [openKpi, setOpenKpi] = useState(null);
  const [openRank, setOpenRank] = useState(null);
  // 판매 급증 TOP5 카드의 정렬 기준('qty'|'pct') — 상세보기 모달을 열 때 그대로 넘겨줘서
  // 모달 표 기본 정렬도 카드에서 고른 기준과 일치하게 한다.
  const [surgeSortMode, setSurgeSortMode] = useState('qty');
  // 권역별 매출 현황 상세보기 모달의 뷰 모드('table'|'map') — 모달 헤더 토글이 제어한다.
  const [regionViewMode, setRegionViewMode] = useState('table');
  // 상단 KPI 3개 카드가 공유하는 스파크라인 조회 기간 — 한 카드에서 토글하면 셋 다 같이 바뀐다.
  const [sparkRangeDays, setSparkRangeDays] = useState(30);
  const toggleSparkRange = () => setSparkRangeDays((d) => (d === 30 ? 7 : 30));

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
          key: 'sales', label: '판매금액', color: '#3b82f6',
          value: fmtWon(summary.total_sales_amount),
          changePct: salesChange.pct,
          changeAbsText: salesChange.abs === null ? null : `${salesChange.abs >= 0 ? '+' : ''}${fmtWon(salesChange.abs)}`,
          wowPct: computeWowChange(summary.sparkline, 'sales_amount').pct,
          sparkData: buildSparkData(summary.sparkline, 'sales_amount', sparkRangeDays),
          sparkValueFormat: fmtWon,
        },
        {
          key: 'count', label: '판매건수', color: '#10b981',
          value: fmtNum(summary.sales_record_count),
          changePct: countChange.pct,
          changeAbsText: countChange.abs === null ? null : `${signed(Math.round(countChange.abs))}건`,
          wowPct: computeWowChange(summary.sparkline, 'sales_record_count').pct,
          sparkData: buildSparkData(summary.sparkline, 'sales_record_count', sparkRangeDays),
          sparkValueFormat: (v) => `${fmtNum(v)}건`,
        },
        {
          key: 'sku', label: '판매 SKU 수', color: '#8b5cf6',
          value: fmtNum(summary.active_sku_count),
          changePct: skuChange.pct,
          changeAbsText: skuChange.abs === null ? null : `${signed(Math.round(skuChange.abs))}개`,
          wowPct: computeWowChange(summary.sparkline, 'active_sku_count').pct,
          sparkData: buildSparkData(summary.sparkline, 'active_sku_count', sparkRangeDays),
          sparkValueFormat: (v) => `${fmtNum(v)}개`,
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
          key: 'sales', label: '판매금액', color: '#3b82f6',
          value: fmtWon(summary.total_sales_amount),
          changePct: salesChange.pct,
          changeAbsText: salesChange.abs === null ? null : `${salesChange.abs >= 0 ? '+' : ''}${fmtWon(salesChange.abs)}`,
          wowPct: computeWowChange(summary.sparkline, 'sales_amount').pct,
          sparkData: buildSparkData(summary.sparkline, 'sales_amount', sparkRangeDays),
          sparkValueFormat: fmtWon,
        },
        {
          key: 'qty', label: '판매수량', color: '#10b981',
          value: `${fmtNum(skuSalesQty)} ${effectiveUnit}`,
          changePct: qtyChange.pct,
          changeAbsText: qtyChange.abs === null ? null : `${signed(Math.round(qtyChange.abs))} ${effectiveUnit}`,
          // 해당 SKU만의 일별 판매수량 시계열은 요약 API에 없어(daily/summary sparkline은
          // 금액/건수/SKU수 기준), 근사치로 순판매금액 추이를 대신 보여준다.
          wowPct: computeWowChange(summary.sparkline, 'net_sales_amount').pct,
          sparkData: buildSparkData(summary.sparkline, 'net_sales_amount', sparkRangeDays),
          sparkValueFormat: fmtWon,
        },
        {
          key: 'count', label: '판매건수', color: '#8b5cf6',
          value: fmtNum(summary.sales_record_count),
          changePct: countChange.pct,
          changeAbsText: countChange.abs === null ? null : `${signed(Math.round(countChange.abs))}건`,
          wowPct: computeWowChange(summary.sparkline, 'sales_record_count').pct,
          sparkData: buildSparkData(summary.sparkline, 'sales_record_count', sparkRangeDays),
          sparkValueFormat: (v) => `${fmtNum(v)}건`,
        },
      ];
    }
    // 3개 카드가 같은 기간 토글 state를 공유하도록 공통 props를 일괄로 얹는다.
    kpis = kpis.map((k) => ({ ...k, sparkRangeDays, onToggleSparkRange: toggleSparkRange }));
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

  const prevReturnAmount = prevDayTx ? prevDayTx.return_amount : null;
  const returnAmountDayChange = (todayTx && prevDayTx) ? computeChange(todayTx.return_amount, prevDayTx.return_amount) : { pct: null, abs: null };

  const gridStyle = { display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '12px' };

  return (
    <>
      <div className="dm-page-hd">
        <h2>{categoryLabel || (selectedSku ? selectedSku.product_name : '전체')}</h2>
        {selectedSku ? (
          // TXT_상품안내_선택 — 상품이 이미 선택된 상태라 "선택하세요" 유도 문구는 빼고,
          // 어떤 상품을 보고 있는지 + 변경/해제 방법만 안내한다.
          <p>
            {center}센터 · {selectedSku.product_name} ({selectedSku.option_code}) 상품 기준 데이터입니다.
            좌측 사이드바에서 다른 상품으로 변경하거나 선택을 해제할 수 있습니다.
          </p>
        ) : (
          // TXT_상품안내_전체
          <p>
            {center}센터 {categoryLabel ? `· ${categoryLabel} ` : ''}
            기준 데이터입니다. 카테고리 또는 상품을 선택하여 더 자세한 정보를 확인할 수 있습니다.
          </p>
        )}
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
            selectedPath={selectedPath}
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
            {/* 집계 모드에서만 조회 기간 컨트롤 표시 — SKU 선택 시 forecast 응답이 자체 history 포함 */}
            {!hasSku && (
              <HistoryRangeControl
                operationalDate={operationalDate}
                historyWeeks={historyWeeks}
                onChangeWeeks={onHistoryWeeksChange}
              />
            )}
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
            todayReturnAmount={todayReturnAmount}
            prevReturnAmount={prevReturnAmount}
            returnAmountDayPct={returnAmountDayChange.pct}
            returnAmountDayAbsText={returnAmountDayChange.abs === null ? null : `${returnAmountDayChange.abs >= 0 ? '+' : ''}${fmtWon(returnAmountDayChange.abs)}`}
          />
        )}

        {/* ── Row 3: 4칸 ── */}
        {!hasSku ? (
          <>
            <AiTop5Card data={aiTop5} loading={aiTop5Loading} error={aiTop5Error} onOpenDetail={() => setOpenRank('ai')} />
            <ShortageTop5Card data={shortage} loading={shortageLoading} error={shortageError} onOpenDetail={() => setOpenRank('shortage')} />
            <SurgeTop5Card
              data={surge} loading={surgeLoading} error={surgeError}
              sortMode={surgeSortMode} onChangeSortMode={setSurgeSortMode}
              onOpenDetail={() => setOpenRank('surge')}
            />
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
          center={center} operationalDate={operationalDate} categoryLabel={categoryLabel} unit={effectiveUnit}
          categoryLarge={large} categoryMiddle={middle} categorySmall={small}
          onClose={() => setOpenKpi(null)}
        />
      )}

      {openRank && (
        <InsightModal
          title={openRank === 'category' ? categoryDrilldownInfo(selectedPath).title : INSIGHT_TITLES[openRank]}
          center={center}
          dateLabel={(openRank === 'ai' || openRank === 'shortage') ? '기준주' : '조회일'}
          basisWeek={(openRank === 'ai' || openRank === 'shortage') ? aiBasisWeek : operationalDate}
          wide={openRank === 'ai' || openRank === 'region'}
          headerExtra={openRank === 'region' ? <RegionViewToggle value={regionViewMode} onChange={setRegionViewMode} /> : null}
          onClose={() => setOpenRank(null)}
        >
          {openRank === 'category' && categorySales && (
            <CategoryOrProductDetail
              data={categorySales} selectedPath={selectedPath}
              onSelectCategory={(label) => { onSelectCategoryFromRanking(label); setOpenRank(null); }}
            />
          )}
          {openRank === 'region' && regionSales && <RegionDetail data={regionSales} viewMode={regionViewMode} />}
          {openRank === 'ai' && <AiTop5Detail data={aiTop5} />}
          {openRank === 'shortage' && shortage && <ShortageDetail data={shortage} />}
          {openRank === 'surge' && surge && <SurgeDetail data={surge} initialSortMode={surgeSortMode} />}
          {openRank === 'returns' && <ReturnQtyDetail data={returnsRanking} />}
        </InsightModal>
      )}
    </>
  );
}
