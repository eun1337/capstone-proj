import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client.js';
import CategorySidebar from '../components/CategorySidebar.jsx';
import ProductPickerModal from '../components/ProductPickerModal.jsx';
import DashboardMain from './DashboardMain.jsx';
import './Dashboard.css';

// 대시보드 운영정보(KPI/차트/카테고리·지역/판매증가/반품)는 operationalDate 기준 일간이고,
// AI TOP5/재고부족/수요예측 추이의 예측 부분은 operationalDate 시점에 이미 존재했던 가장
// 최근 forecast 원점(주간, aiBasisWeek)을 쓴다 — 두 시간축을 하나로 합치지 않는다.
const DEFAULT_OPERATIONAL_DATE = '2024-09-30';
// 보유 데이터 범위(실제 관측된 마지막 일자는 2024-12-31) — "실시간"이 아니라 이 범위까지만 조회 가능.
const OPERATIONAL_DATE_MIN = '2024-01-01';
const OPERATIONAL_DATE_MAX = '2024-12-31';
const UNITS = ['EA', 'BX', 'CS'];

const TABS = [
  { key: 'dashboard', label: '대시보드' },
  { key: 'tableau', label: 'Tableau' },
];

function fmtISODate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function shiftDate(iso, days) {
  const d = new Date(`${iso}T00:00:00`);
  d.setDate(d.getDate() + days);
  return fmtISODate(d);
}

// 일간 조회일(operationalDate) 이하 가장 최근 월요일 — weekly /inventory 조회 전용.
// forecast_basis_week(=forecast_2024 원본이 실제로 존재하는 주)와는 다른 개념이다: 이건 순수
// 달력 계산이고, weekly_demand/inventory_weekly는 모든 월요일에 대해 dense grid이므로
// forecast 유무와 무관하게 항상 유효한 기준주를 준다.
function mostRecentMonday(iso) {
  const d = new Date(`${iso}T00:00:00`);
  const diff = (d.getDay() + 6) % 7;
  d.setDate(d.getDate() - diff);
  return fmtISODate(d);
}

function formatDateLabel(iso) {
  const d = new Date(`${iso}T00:00:00`);
  const days = ['일', '월', '화', '수', '목', '금', '토'];
  return `${iso} (${days[d.getDay()]})`;
}

export default function Dashboard() {
  const [center, setCenter] = useState('A');
  const [operationalDate, setOperationalDate] = useState(DEFAULT_OPERATIONAL_DATE);
  // 수량 기반 카드(수요예측 추이/AI TOP5/재고부족/판매증가/반품수량) 공통 단위. EA/BX/CS만
  // 있고 "전체"는 없다 — 서로 다른 barcode의 별도 SKU라 합산/환산하지 않는다.
  const [topUnit, setTopUnit] = useState('EA');
  const [activeTab, setActiveTab] = useState('dashboard');

  const [categoryTree, setCategoryTree] = useState([]);
  const [categoriesLoading, setCategoriesLoading] = useState(true);
  const [categoriesError, setCategoriesError] = useState(null);
  const [openMap, setOpenMap] = useState({});
  const [selectedPath, setSelectedPath] = useState([]);
  const [categorySearch, setCategorySearch] = useState('');

  const [selectedSku, setSelectedSku] = useState(null); // 상품 찾기 modal에서 선택 완료한 SKU
  const [showProductPicker, setShowProductPicker] = useState(false);

  const [large, middle, small] = selectedPath;
  const categoryLabel = selectedPath.length > 0 ? selectedPath.join(' > ') : null;
  // 상품이 선택되면 상단 단위는 그 SKU의 실제 option_code로 고정된다.
  const effectiveUnit = selectedSku ? selectedSku.option_code : topUnit;
  const prevDate = shiftDate(operationalDate, -1);
  const prevWeekDate = shiftDate(operationalDate, -7);

  const navigate = useNavigate();
  const username = localStorage.getItem('username') || 'admin';

  // ── 카테고리 트리 ──────────────────────────────────────────────
  useEffect(() => {
    let ignore = false;
    setCategoriesLoading(true);
    setCategoriesError(null);
    api.getCategories({ center })
      .then((data) => { if (!ignore) setCategoryTree(data); })
      .catch((e) => { if (!ignore) setCategoriesError(e.message); })
      .finally(() => { if (!ignore) setCategoriesLoading(false); });
    return () => { ignore = true; };
  }, [center]);

  // ── KPI 요약(오늘/전일) ────────────────────────────────────────
  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState(null);
  const [prevSummary, setPrevSummary] = useState(null);

  useEffect(() => {
    let ignore = false;
    setSummaryLoading(true);
    setSummaryError(null);
    const params = {
      center, category_large: large, category_middle: middle, category_small: small,
      sku_id: selectedSku?.sku_id,
    };
    Promise.all([
      api.getDailySummary({ ...params, date: operationalDate }),
      api.getDailySummary({ ...params, date: prevDate }).catch(() => null),
    ])
      .then(([today, prev]) => { if (!ignore) { setSummary(today); setPrevSummary(prev); } })
      .catch((e) => { if (!ignore) { setSummaryError(e.message); setSummary(null); setPrevSummary(null); } })
      .finally(() => { if (!ignore) setSummaryLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, selectedSku?.sku_id, operationalDate, prevDate]);

  const aiBasisWeek = summary?.forecast_basis_week ?? null;
  const inventoryBasisWeek = mostRecentMonday(operationalDate);

  // ── 집계 모드(SKU 미선택) 전용 데이터 ──────────────────────────
  const [categorySales, setCategorySales] = useState(null);
  const [categorySalesLoading, setCategorySalesLoading] = useState(true);
  const [categorySalesError, setCategorySalesError] = useState(null);

  const [regionSales, setRegionSales] = useState(null);
  const [regionSalesLoading, setRegionSalesLoading] = useState(true);
  const [regionSalesError, setRegionSalesError] = useState(null);

  const [aiTop5, setAiTop5] = useState([]);
  const [aiTop5Loading, setAiTop5Loading] = useState(true);
  const [aiTop5Error, setAiTop5Error] = useState(null);

  const [shortage, setShortage] = useState(null);
  const [shortageLoading, setShortageLoading] = useState(true);
  const [shortageError, setShortageError] = useState(null);

  const [surge, setSurge] = useState(null);
  const [surgeLoading, setSurgeLoading] = useState(true);
  const [surgeError, setSurgeError] = useState(null);

  const [returnsRanking, setReturnsRanking] = useState([]);
  const [returnsLoading, setReturnsLoading] = useState(true);
  const [returnsError, setReturnsError] = useState(null);

  const [demandTrend, setDemandTrend] = useState(null);
  const [demandTrendLoading, setDemandTrendLoading] = useState(true);
  const [demandTrendError, setDemandTrendError] = useState(null);

  useEffect(() => {
    if (selectedSku) return;
    let ignore = false;
    setCategorySalesLoading(true);
    setCategorySalesError(null);
    api.getDailyCategorySales({ center, date: operationalDate, category_large: large, category_middle: middle, category_small: small })
      .then((data) => { if (!ignore) setCategorySales(data); })
      .catch((e) => { if (!ignore) { setCategorySalesError(e.message); setCategorySales(null); } })
      .finally(() => { if (!ignore) setCategorySalesLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, operationalDate, selectedSku]);

  useEffect(() => {
    if (selectedSku) return;
    let ignore = false;
    setRegionSalesLoading(true);
    setRegionSalesError(null);
    api.getDailyRegionSales({ center, date: operationalDate, category_large: large, category_middle: middle, category_small: small })
      .then((data) => { if (!ignore) setRegionSales(data); })
      .catch((e) => { if (!ignore) { setRegionSalesError(e.message); setRegionSales(null); } })
      .finally(() => { if (!ignore) setRegionSalesLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, operationalDate, selectedSku]);

  useEffect(() => {
    if (selectedSku || !aiBasisWeek) {
      setAiTop5([]);
      setAiTop5Error(null);
      setAiTop5Loading(!aiBasisWeek && !selectedSku);
      return;
    }
    let ignore = false;
    setAiTop5Loading(true);
    setAiTop5Error(null);
    api.getForecastProducts({
      center, week_st: aiBasisWeek, category_large: large, category_middle: middle, category_small: small,
      option_code: topUnit,
    })
      .then((data) => { if (!ignore) setAiTop5(data); })
      .catch((e) => { if (!ignore) { setAiTop5Error(e.message); setAiTop5([]); } })
      .finally(() => { if (!ignore) setAiTop5Loading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, aiBasisWeek, topUnit, selectedSku]);

  useEffect(() => {
    if (selectedSku || !aiBasisWeek) {
      setShortage(null);
      setShortageError(null);
      setShortageLoading(!aiBasisWeek && !selectedSku);
      return;
    }
    let ignore = false;
    setShortageLoading(true);
    setShortageError(null);
    api.getInsightInventoryShortage({
      center, week_st: aiBasisWeek, category_large: large, category_middle: middle, category_small: small,
      option_code: topUnit,
    })
      .then((data) => { if (!ignore) setShortage(data); })
      .catch((e) => { if (!ignore) { setShortageError(e.message); setShortage(null); } })
      .finally(() => { if (!ignore) setShortageLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, aiBasisWeek, topUnit, selectedSku]);

  useEffect(() => {
    if (selectedSku) return;
    let ignore = false;
    setSurgeLoading(true);
    setSurgeError(null);
    api.getDailySalesSurge({
      center, date: operationalDate, category_large: large, category_middle: middle, category_small: small,
      option_code: topUnit,
    })
      .then((data) => { if (!ignore) setSurge(data); })
      .catch((e) => { if (!ignore) { setSurgeError(e.message); setSurge(null); } })
      .finally(() => { if (!ignore) setSurgeLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, operationalDate, topUnit, selectedSku]);

  // 오늘 반품수량 TOP5 — 기존 /daily/returns(금액 정렬) 전체 ranking을 그대로 받아 상단
  // 단위(topUnit)로 필터링하고 반품수량 내림차순으로만 클라이언트에서 재정렬한다(새 backend 없음).
  useEffect(() => {
    if (selectedSku) return;
    let ignore = false;
    setReturnsLoading(true);
    setReturnsError(null);
    api.getDailyReturns({ center, date: operationalDate, category_large: large, category_middle: middle, category_small: small })
      .then((data) => {
        if (ignore) return;
        const sorted = (data.ranking || [])
          .filter((r) => r.option_code === topUnit)
          .sort((a, b) => b.return_qty - a.return_qty);
        setReturnsRanking(sorted);
      })
      .catch((e) => { if (!ignore) { setReturnsError(e.message); setReturnsRanking([]); } })
      .finally(() => { if (!ignore) setReturnsLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, operationalDate, topUnit, selectedSku]);

  useEffect(() => {
    if (selectedSku || !aiBasisWeek) {
      setDemandTrend(null);
      setDemandTrendError(null);
      setDemandTrendLoading(!aiBasisWeek && !selectedSku);
      return;
    }
    let ignore = false;
    setDemandTrendLoading(true);
    setDemandTrendError(null);
    api.getDemandTrend({
      center, option_code: topUnit, category_large: large, category_middle: middle, category_small: small,
      week_st: aiBasisWeek, history_weeks: 12,
    })
      .then((data) => { if (!ignore) setDemandTrend(data); })
      .catch((e) => { if (!ignore) { setDemandTrendError(e.message); setDemandTrend(null); } })
      .finally(() => { if (!ignore) setDemandTrendLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, aiBasisWeek, topUnit, selectedSku]);

  // ── SKU 선택 모드 전용 데이터 ──────────────────────────────────
  const [forecast, setForecast] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError] = useState(null);

  const [inventory, setInventory] = useState(null);
  const [inventoryLoading, setInventoryLoading] = useState(false);
  const [inventoryError, setInventoryError] = useState(null);

  const [transactions, setTransactions] = useState(null);
  const [transactionsLoading, setTransactionsLoading] = useState(false);
  const [transactionsError, setTransactionsError] = useState(null);

  useEffect(() => {
    if (!selectedSku || !aiBasisWeek) {
      setForecast(null);
      setForecastError(null);
      setForecastLoading(false);
      return;
    }
    let ignore = false;
    setForecastLoading(true);
    setForecastError(null);
    api.getDashboardForecast({ center: selectedSku.center_id, sku_id: selectedSku.sku_id, week_st: aiBasisWeek })
      .then((data) => { if (!ignore) setForecast(data); })
      .catch((e) => { if (!ignore) { setForecastError(e.message); setForecast(null); } })
      .finally(() => { if (!ignore) setForecastLoading(false); });
    return () => { ignore = true; };
  }, [selectedSku, aiBasisWeek]);

  useEffect(() => {
    if (!selectedSku) {
      setInventory(null);
      setInventoryError(null);
      setInventoryLoading(false);
      return;
    }
    let ignore = false;
    setInventoryLoading(true);
    setInventoryError(null);
    api.getDashboardInventory({
      center: selectedSku.center_id, sku_id: selectedSku.sku_id, week_st: inventoryBasisWeek, return_policy: 'excluded',
    })
      .then((data) => { if (!ignore) setInventory(data); })
      .catch((e) => { if (!ignore) { setInventoryError(e.message); setInventory(null); } })
      .finally(() => { if (!ignore) setInventoryLoading(false); });
    return () => { ignore = true; };
  }, [selectedSku, inventoryBasisWeek]);

  useEffect(() => {
    if (!selectedSku) {
      setTransactions(null);
      setTransactionsError(null);
      setTransactionsLoading(false);
      return;
    }
    let ignore = false;
    setTransactionsLoading(true);
    setTransactionsError(null);
    api.getDailyTransactions({ center: selectedSku.center_id, sku_id: selectedSku.sku_id, date: operationalDate, history_days: 30 })
      .then((data) => { if (!ignore) setTransactions(data); })
      .catch((e) => { if (!ignore) { setTransactionsError(e.message); setTransactions(null); } })
      .finally(() => { if (!ignore) setTransactionsLoading(false); });
    return () => { ignore = true; };
  }, [selectedSku, operationalDate]);

  // ── 핸들러 ─────────────────────────────────────────────────────
  function handleCenterChange(next) {
    if (next === center) return;
    setCenter(next);
    setSelectedPath([]);
    setSelectedSku(null);
    setOpenMap({});
  }

  function handleSelectCategory(path) {
    setSelectedPath(path);
    setSelectedSku(null);
  }

  function handleToggleCategory(key) {
    setOpenMap((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function handleClearSku() {
    setSelectedSku(null);
  }

  function handleConfirmProduct(product) {
    setSelectedSku(product);
    setTopUnit(product.option_code);
    setShowProductPicker(false);
  }

  function handleSelectCategoryFromRanking(label) {
    setSelectedPath([...selectedPath, label]);
    setSelectedSku(null);
  }

  function handleLogout() {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    navigate('/');
  }

  return (
    <div className="dash-root">
      <header className="dash-header">
        <div className="header-brand">
          <span className="brand-icon">📦</span>
          <span className="brand-name">IN-SIGHT</span>
        </div>
        <nav className="header-nav">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`nav-item ${activeTab === t.key ? 'active' : ''}`}
              onClick={() => setActiveTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="header-actions">
          <div className="header-center-toggle">
            <button className={`center-btn ${center === 'A' ? 'active' : ''}`} onClick={() => handleCenterChange('A')}>A센터</button>
            <button className={`center-btn ${center === 'B' ? 'active' : ''}`} onClick={() => handleCenterChange('B')}>B센터</button>
          </div>
          <div className="basis-week-picker">
            <label htmlFor="operational-date-input" className="basis-week-label">기준일</label>
            <input
              id="operational-date-input"
              type="date"
              className="operational-date-input"
              value={operationalDate}
              min={OPERATIONAL_DATE_MIN}
              max={OPERATIONAL_DATE_MAX}
              onChange={(e) => e.target.value && setOperationalDate(e.target.value)}
            />
          </div>
          <div className="basis-week-picker">
            <label htmlFor="top-unit-select" className="basis-week-label">단위</label>
            <select
              id="top-unit-select"
              className="basis-week-select"
              value={effectiveUnit}
              disabled={Boolean(selectedSku)}
              onChange={(e) => setTopUnit(e.target.value)}
            >
              {UNITS.map((u) => <option key={u} value={u}>{u}</option>)}
            </select>
          </div>
          <div className="user-chip">
            <span>👤</span>
            <span>{username}</span>
          </div>
          <button className="hdr-logout" onClick={handleLogout}>로그아웃</button>
        </div>
      </header>

      <div className="main-body">
        <CategorySidebar
          categoryTree={categoryTree} categoriesLoading={categoriesLoading} categoriesError={categoriesError}
          selectedPath={selectedPath} onSelectCategory={handleSelectCategory}
          openMap={openMap} onToggleCategory={handleToggleCategory}
          categorySearch={categorySearch} onCategorySearchChange={setCategorySearch}
          selectedSku={selectedSku} onOpenProductPicker={() => setShowProductPicker(true)} onClearSku={handleClearSku}
        />

        <div className="content-area">
          {activeTab === 'dashboard' && (
            <DashboardMain
              center={center} operationalDate={operationalDate} prevDate={prevDate} prevWeekDate={prevWeekDate}
              formatDateLabel={formatDateLabel}
              topUnit={topUnit} effectiveUnit={effectiveUnit}
              selectedPath={selectedPath} categoryLabel={categoryLabel}
              selectedSku={selectedSku}
              aiBasisWeek={aiBasisWeek} inventoryBasisWeek={inventoryBasisWeek}
              summary={summary} summaryLoading={summaryLoading} summaryError={summaryError} prevSummary={prevSummary}
              categorySales={categorySales} categorySalesLoading={categorySalesLoading} categorySalesError={categorySalesError}
              regionSales={regionSales} regionSalesLoading={regionSalesLoading} regionSalesError={regionSalesError}
              aiTop5={aiTop5} aiTop5Loading={aiTop5Loading} aiTop5Error={aiTop5Error}
              shortage={shortage} shortageLoading={shortageLoading} shortageError={shortageError}
              surge={surge} surgeLoading={surgeLoading} surgeError={surgeError}
              returnsRanking={returnsRanking} returnsLoading={returnsLoading} returnsError={returnsError}
              demandTrend={demandTrend} demandTrendLoading={demandTrendLoading} demandTrendError={demandTrendError}
              forecast={forecast} forecastLoading={forecastLoading} forecastError={forecastError}
              inventory={inventory} inventoryLoading={inventoryLoading} inventoryError={inventoryError}
              transactions={transactions} transactionsLoading={transactionsLoading} transactionsError={transactionsError}
              onSelectCategoryFromRanking={handleSelectCategoryFromRanking}
            />
          )}
          {activeTab === 'tableau' && <TableauView onBack={() => setActiveTab('dashboard')} />}
        </div>
      </div>

      {showProductPicker && (
        <ProductPickerModal
          center={center} selectedPath={selectedPath} categoryLabel={categoryLabel}
          operationalDate={operationalDate} aiBasisWeek={aiBasisWeek} initialUnit={topUnit}
          onCancel={() => setShowProductPicker(false)}
          onConfirm={handleConfirmProduct}
        />
      )}
    </div>
  );
}

// ── Tableau 연동 영역 (기존 로직 보존, 절대 수정하지 않음) ────────────────────
function TableauView({ onBack }) {
  const src = 'https://public.tableau.com/views/YOUR_WORKBOOK/DemandForecast';
  return (
    <div className="tableau-view">
      <div className="tableau-view-hd">
        <button className="back-btn" onClick={onBack}>← 대시보드로</button>
        <h2>Tableau 상세 분석</h2>
        <span className="tableau-badge">JWT: GET /api/tableau-token</span>
      </div>
      {/*
        실제 연동 시:
        1. frontend/index.html Tableau Embedding API 스크립트 주석 해제
        2. GET /api/tableau-token 으로 JWT 발급 후 token 속성에 전달
        3. src 를 실제 워크북 URL 로 교체

        <tableau-viz
          src={src}
          token="{jwt}"
          device="desktop"
          hide-tabs
          toolbar="hidden"
        />
      */}
      <div className="tableau-ph">
        <div className="tableau-ph-inner">
          <div className="tableau-ph-icon">📊</div>
          <h3>Tableau 대시보드 연동 영역</h3>
          <p>Tableau Connected App 설정 완료 후 실제 뷰가 이 영역에 표시됩니다.</p>
          <div className="tableau-ph-rows">
            <div className="tableau-ph-row">
              <span>JWT 발급</span><code>GET /api/tableau-token</code>
            </div>
            <div className="tableau-ph-row">
              <span>Tableau URL</span><code>{src}</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
