import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import KpiCard from '../components/KpiCard.jsx';
import ForecastChart from '../components/ForecastChart.jsx';
import DonutChart from '../components/DonutChart.jsx';
import BottomGrid from '../components/BottomGrid.jsx';
import './Dashboard.css';

// ── KPI Mock Data ──────────────────────────────────────────────
const KPI_LIST = [
  { key: 'sales',    label: 'Sales',             value: '₩1.24B', change: +8.3,  color: '#3b82f6',
    sparkData: [42, 55, 48, 60, 53, 65, 72, 68, 75, 82, 79, 88] },
  { key: 'orders',   label: 'Number of Orders',  value: '24,182', change: +12.1, color: '#8b5cf6',
    sparkData: [30, 35, 28, 40, 38, 45, 42, 50, 47, 55, 52, 60] },
  { key: 'orderQty', label: 'Order Qty',          value: '87,430', change: -2.4,  color: '#f59e0b',
    sparkData: [80, 75, 82, 70, 78, 73, 68, 72, 65, 70, 67, 63] },
  { key: 'aov',      label: 'AOV',                value: '₩51,300', change: +5.7, color: '#10b981',
    sparkData: [44, 46, 48, 47, 49, 50, 51, 49, 52, 53, 51, 54] },
  { key: 'asp',      label: 'ASP',                value: '₩14,200', change: -1.2, color: '#ef4444',
    sparkData: [15, 14.5, 14.8, 14.2, 14.6, 14.3, 14.5, 14.1, 14.3, 14.0, 14.2, 14.1] },
];

// ── KAN 표준 분류 트리 ─────────────────────────────────────────
const CATEGORY_TREE = [
  {
    id: 'processed', label: '가공식품', icon: '🛒',
    children: [
      { id: 'ramen', label: '라면', children: [
          { id: 'bag-ramen', label: '봉지라면' },
          { id: 'cup-ramen', label: '용기라면' },
        ],
      },
      { id: 'snacks',    label: '과자류' },
      { id: 'beverages', label: '음료' },
      { id: 'canned',    label: '통조림/즉석식품' },
    ],
  },
  {
    id: 'fresh', label: '신선식품', icon: '🥬',
    children: [
      { id: 'fruits',     label: '과일' },
      { id: 'vegetables', label: '채소' },
      { id: 'meat',       label: '육류' },
      { id: 'seafood',    label: '수산물' },
    ],
  },
  {
    id: 'grain', label: '곡물', icon: '🌾',
    children: [
      { id: 'rice',  label: '쌀' },
      { id: 'flour', label: '밀가루/잡곡' },
    ],
  },
  {
    id: 'daily', label: '일상용품', icon: '🧴',
    children: [
      { id: 'household', label: '생활용품' },
      { id: 'detergent', label: '세제류' },
      { id: 'personal',  label: '개인위생' },
    ],
  },
];

// ── Accordion Node ─────────────────────────────────────────────
function CategoryNode({ node, depth = 0, selected, onSelect, openMap, onToggle }) {
  const isOpen     = !!openMap[node.id];
  const hasKids    = node.children?.length > 0;
  const isSelected = selected === node.id;

  return (
    <div>
      <button
        className={`cat-btn depth-${depth} ${isSelected ? 'cat-active' : ''}`}
        onClick={() => { if (hasKids) onToggle(node.id); onSelect(node.id); }}
      >
        <span className="cat-label-wrap">
          {depth === 0 && node.icon && <span className="cat-icon">{node.icon}</span>}
          <span>{node.label}</span>
        </span>
        {hasKids && <span className={`cat-chevron ${isOpen ? 'open' : ''}`}>›</span>}
      </button>
      {hasKids && isOpen && (
        <div>
          {node.children.map((c) => (
            <CategoryNode
              key={c.id} node={c} depth={depth + 1}
              selected={selected} onSelect={onSelect}
              openMap={openMap} onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Dashboard ─────────────────────────────────────────────────
export default function Dashboard() {
  const [openMap,      setOpenMap]      = useState({ processed: true });
  const [selectedCat,  setSelectedCat]  = useState(null);
  const [forecastDays, setForecastDays] = useState(30);
  const [showTableau,  setShowTableau]  = useState(false);
  const navigate = useNavigate();
  const username = localStorage.getItem('username') || 'admin';
  const today = new Date().toLocaleDateString('ko-KR', {
    year: 'numeric', month: '2-digit', day: '2-digit',
  });

  function toggleCategory(id) {
    setOpenMap((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  function handleLogout() {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    navigate('/');
  }

  return (
    <div className="dash-root">
      {/* ── Header ── */}
      <header className="dash-header">
        <div className="header-brand">
          <span className="brand-icon">📦</span>
          <span className="brand-name">LogiSense</span>
        </div>
        <nav className="header-nav">
          <button className="nav-item active">대시보드</button>
          <button className="nav-item">수요예측</button>
          <button className="nav-item">재고관리</button>
          <button className="nav-item" onClick={() => setShowTableau((v) => !v)}>
            {showTableau ? '← 대시보드' : 'Tableau'}
          </button>
        </nav>
        <div className="header-actions">
          <span className="header-date">{today}</span>
          <div className="user-chip">
            <span>👤</span>
            <span>{username}</span>
          </div>
          <button className="hdr-logout" onClick={handleLogout}>로그아웃</button>
        </div>
      </header>

      {/* ── KPI Row ── */}
      <div className="kpi-row">
        {KPI_LIST.map((kpi) => <KpiCard key={kpi.key} {...kpi} />)}
      </div>

      {/* ── Main Body ── */}
      <div className="main-body">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-hd">KAN 표준 분류</div>
          <div className="sidebar-search">
            <input type="text" placeholder="카테고리 검색..." />
          </div>
          <div className="cat-tree">
            {CATEGORY_TREE.map((node) => (
              <CategoryNode
                key={node.id} node={node}
                selected={selectedCat} onSelect={setSelectedCat}
                openMap={openMap} onToggle={toggleCategory}
              />
            ))}
          </div>
        </aside>

        {/* Content */}
        <div className="content-area">
          {showTableau ? (
            <TableauView onBack={() => setShowTableau(false)} />
          ) : (
            <>
              {/* Charts */}
              <div className="charts-row">
                <div className="chart-card forecast-card">
                  <div className="chart-card-hd">
                    <div>
                      <h3>센터 전체 수요예측 추이</h3>
                      <p>실선: 실제 판매량 &nbsp;·&nbsp; 점선: AI 예측</p>
                    </div>
                    <div className="forecast-btns">
                      {[7, 14, 30].map((d) => (
                        <button
                          key={d}
                          className={`filter-btn ${forecastDays === d ? 'active' : ''}`}
                          onClick={() => setForecastDays(d)}
                        >
                          {d}일 예측
                        </button>
                      ))}
                    </div>
                  </div>
                  <ForecastChart forecastDays={forecastDays} />
                </div>

                <div className="chart-card donut-card">
                  <div className="chart-card-hd">
                    <div>
                      <h3>재고 위험도 현황</h3>
                      <p>전체 품목 재고 상태 분포</p>
                    </div>
                  </div>
                  <DonutChart />
                </div>
              </div>

              {/* Bottom Grid */}
              <BottomGrid />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Tableau 연동 영역 (기존 로직 보존) ────────────────────────
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
