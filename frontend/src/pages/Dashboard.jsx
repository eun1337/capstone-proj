import { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { api } from '../api/client.js';
import CategorySidebar from '../components/CategorySidebar.jsx';
import ProductPickerModal from '../components/ProductPickerModal.jsx';
import DashboardMain from './DashboardMain.jsx';
import './Dashboard.css';

const DEFAULT_OPERATIONAL_DATE = '2024-09-30';
const OPERATIONAL_DATE_MIN = '2024-01-01';
const OPERATIONAL_DATE_MAX = '2024-12-31';
const UNITS = ['EA', 'BX', 'CS'];

// to가 있는 탭은 별도 라우트(/analysis/*)로 이동하고, 없는 탭은 이 페이지 안에서 전환한다.
const TABS = [
  { key: 'dashboard', label: '대시보드' },
  { key: 'model-analysis', label: '분석 과정', to: '/analysis/overview' },
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
  const location = useLocation();
  const [center, setCenter] = useState('A');
  const [operationalDate, setOperationalDate] = useState(DEFAULT_OPERATIONAL_DATE);
  const [topUnit, setTopUnit] = useState('EA');
  const [activeTab, setActiveTab] = useState(location.state?.tab === 'tableau' ? 'tableau' : 'dashboard');
  const [historyWeeks, setHistoryWeeks] = useState(4);

  const [categoryTree, setCategoryTree] = useState([]);
  const [categoriesLoading, setCategoriesLoading] = useState(true);
  const [categoriesError, setCategoriesError] = useState(null);
  const [openMap, setOpenMap] = useState({});
  const [selectedPath, setSelectedPath] = useState([]);
  const [categorySearch, setCategorySearch] = useState('');

  const [selectedSku, setSelectedSku] = useState(null);
  const [showProductPicker, setShowProductPicker] = useState(false);

  const [large, middle, small] = selectedPath;
  const categoryLabel = selectedPath.length > 0 ? selectedPath.join(' > ') : null;
  const effectiveUnit = selectedSku ? selectedSku.option_code : topUnit;
  const prevDate = shiftDate(operationalDate, -1);
  const prevWeekDate = shiftDate(operationalDate, -7);

  const navigate = useNavigate();
  const username = localStorage.getItem('username') || 'admin';

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
      week_st: aiBasisWeek, history_weeks: historyWeeks,
    })
      .then((data) => { if (!ignore) setDemandTrend(data); })
      .catch((e) => { if (!ignore) { setDemandTrendError(e.message); setDemandTrend(null); } })
      .finally(() => { if (!ignore) setDemandTrendLoading(false); });
    return () => { ignore = true; };
  }, [center, large, middle, small, aiBasisWeek, topUnit, selectedSku, historyWeeks]);

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

  function handleLogoClick() {
    setActiveTab('dashboard');
    setCenter('A');
    setOperationalDate(DEFAULT_OPERATIONAL_DATE);
    setTopUnit('EA');
    setHistoryWeeks(4);
    setSelectedPath([]);
    setOpenMap({});
    setCategorySearch('');
    setSelectedSku(null);
    setShowProductPicker(false);
  }

  return (
    <div className="dash-root">
      <header className="dash-header">
        <button type="button" className="header-brand" onClick={handleLogoClick} title="대시보드 초기 화면으로">
          <span className="brand-icon">📦</span>
          <span className="brand-name">IN-SIGHT</span>
        </button>
        <nav className="header-nav">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`nav-item ${activeTab === t.key ? 'active' : ''}`}
              onClick={() => (t.to ? navigate(t.to) : setActiveTab(t.key))}
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
        {/* Tableau 탭에서는 사이드바를 숨겨 iframe이 전체 너비를 점유한다 */}
        {activeTab !== 'tableau' && (
          <CategorySidebar
            categoryTree={categoryTree} categoriesLoading={categoriesLoading} categoriesError={categoriesError}
            selectedPath={selectedPath} onSelectCategory={handleSelectCategory}
            openMap={openMap} onToggleCategory={handleToggleCategory}
            categorySearch={categorySearch} onCategorySearchChange={setCategorySearch}
            selectedSku={selectedSku} onOpenProductPicker={() => setShowProductPicker(true)} onClearSku={handleClearSku}
          />
        )}

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
              historyWeeks={historyWeeks} onHistoryWeeksChange={setHistoryWeeks}
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

// ── Tableau Connected Apps SSO 임베딩 ─────────────────────────────────────────
//
// 문제: React가 JSX로 <tableau-viz>를 렌더링할 때 Tableau Embedding API 스크립트가
//       아직 커스텀 엘리먼트를 등록하기 전일 수 있음 → 빈 화면만 표시됨.
// 해결: customElements.whenDefined('tableau-viz')로 등록 완료를 기다린 뒤
//       document.createElement로 명령형 마운트.

// 임베딩할 뷰 주소는 frontend/.env 의 VITE_TABLEAU_VIZ_URL 로 주입한다 (.env.example 참고).
const TABLEAU_VIZ_URL = import.meta.env.VITE_TABLEAU_VIZ_URL;

function TableauView({ onBack }) {
  const [token, setToken]       = useState(null);
  const [status, setStatus]     = useState('loading'); // 'loading' | 'ready' | 'error'
  const [errorMsg, setErrorMsg] = useState('');
  const containerRef = useRef(null); // <tableau-viz>가 마운트될 div
  const vizRef       = useRef(null); // <tableau-viz> DOM 엘리먼트 참조 (토큰 갱신용)

  // ── 토큰 fetch (최초 + 자동 갱신 공용) ────────────────────────────────────
  const fetchToken = useCallback(async () => {
    try {
      const data = await api.getTableauToken();
      setToken(data.token);
      setStatus('ready');
      return data.expires_in;
    } catch (e) {
      setErrorMsg(e.message || '토큰 발급에 실패했습니다.');
      setStatus('error');
      return null;
    }
  }, []);

  // ── 마운트 시 토큰 fetch + 자동 갱신 타이머 ───────────────────────────────
  useEffect(() => {
    let timerId = null;
    async function init() {
      const expiresIn = await fetchToken();
      if (!expiresIn) return;
      const refreshMs = Math.max((expiresIn - 60) * 1000, 10_000);
      timerId = setInterval(async () => {
        const data = await api.getTableauToken().catch(() => null);
        if (data && vizRef.current) {
          // Tableau Embedding API가 제공하는 token setter로 조용히 갱신
          vizRef.current.token = data.token;
        }
      }, refreshMs);
    }
    init();
    return () => { if (timerId) clearInterval(timerId); };
  }, [fetchToken]);

  // ── 토큰 준비 완료 → <tableau-viz> 명령형 마운트 ─────────────────────────
  useEffect(() => {
    if (status !== 'ready' || !token || !containerRef.current) return;

    let cancelled = false;

    (async () => {
      if (!TABLEAU_VIZ_URL) {
        setErrorMsg('VITE_TABLEAU_VIZ_URL 환경변수가 설정되지 않았습니다.');
        setStatus('error');
        return;
      }

      // Embedding API 스크립트 로드 완료 대기 (최대 15초)
      try {
        const defined = customElements.whenDefined('tableau-viz');
        const timeout = new Promise((_, rej) =>
          setTimeout(() => rej(new Error('Tableau API 스크립트 로드 타임아웃(15s)')), 15_000)
        );
        await Promise.race([defined, timeout]);
      } catch (e) {
        if (!cancelled) {
          setErrorMsg(e.message);
          setStatus('error');
        }
        return;
      }

      if (cancelled || !containerRef.current) return;

      containerRef.current.innerHTML = '';

      const viz = document.createElement('tableau-viz');
      viz.setAttribute('src', TABLEAU_VIZ_URL);
      viz.setAttribute('token', token);
      viz.setAttribute('toolbar', 'top');
      viz.setAttribute('device', 'desktop');
      // 시트 탭을 보이게 하려면 아래 줄을 삭제하세요
      viz.setAttribute('hide-tabs', '');
      viz.style.cssText = 'width:100%;height:100%;display:block;';

      containerRef.current.appendChild(viz);
      vizRef.current = viz;
    })();

    return () => {
      cancelled = true;
      if (containerRef.current) containerRef.current.innerHTML = '';
      vizRef.current = null;
    };
  }, [status, token]);


  // ── 재시도 ────────────────────────────────────────────────────────────────
  function handleRetry() {
    setStatus('loading');
    setErrorMsg('');
    setToken(null);
    fetchToken();
  }

  return (
    <div className="tableau-view">
      <div className="tableau-view-hd">
        <button className="back-btn" onClick={onBack}>← 대시보드로</button>
        <h2>Tableau 메인 대시보드</h2>
        {status === 'ready' && (
          <span className="tableau-live-badge">● SSO 연결됨</span>
        )}
      </div>

      <div className="tableau-embed-wrap">
        {status === 'loading' && (
          <div className="tableau-loading">
            <div className="tableau-spinner" />
            <span>SSO 인증 중입니다…</span>
          </div>
        )}
        {status === 'error' && (
          <div className="tableau-error">
            <div className="tableau-error-icon">⚠️</div>
            <p>대시보드를 불러올 수 없습니다.</p>
            <p className="tableau-error-sub">{errorMsg}</p>
            <button className="tableau-retry-btn" onClick={handleRetry}>
              다시 시도
            </button>
          </div>
        )}
        {/* <tableau-viz>는 위 useEffect에서 명령형으로 이 div 안에 마운트됨 */}
        <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      </div>
    </div>
  );
}
