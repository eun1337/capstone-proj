import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { IconAward, IconBarChart, IconHome, IconLayers } from './icons.jsx';
import '../Dashboard.css';
import './AnalysisLayout.css';

const TOP_TABS = [
  { key: 'dashboard', label: '대시보드', to: '/dashboard' },
  { key: 'model-analysis', label: '분석 과정', to: '/analysis/overview' },
  { key: 'tableau', label: 'Tableau', to: '/dashboard', state: { tab: 'tableau' } },
];

const NAV_ITEMS = [
  { to: 'overview', Icon: IconHome, label: '01 모델링 개요' },
  { to: 'stat', Icon: IconBarChart, label: '02 통계모델 분석' },
  { to: 'ml-dl', Icon: IconLayers, label: '03 ML/DL 분석' },
  { to: 'comparison', Icon: IconAward, label: '04 최종 모델 비교' },
];

export default function AnalysisLayout() {
  const navigate = useNavigate();
  const username = localStorage.getItem('username') || 'admin';

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
          {TOP_TABS.map((t) => (
            <button
              key={t.key}
              className={`nav-item ${t.key === 'model-analysis' ? 'active' : ''}`}
              onClick={() => navigate(t.to, t.state ? { state: t.state } : undefined)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="header-actions">
          <div className="user-chip">
            <span>👤</span>
            <span>{username}</span>
          </div>
          <button className="hdr-logout" onClick={handleLogout}>로그아웃</button>
        </div>
      </header>

      <div className="main-body">
        <aside className="az-sidebar">
          <nav className="az-sidebar-nav">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `az-nav-item ${isActive ? 'active' : ''}`}
              >
                <item.Icon className="az-nav-icon" />
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>
        </aside>

        <div className="content-area">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
