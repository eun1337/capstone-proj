import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import { hasValidSession } from './api/session.js';
import Login from './pages/Login.jsx';
import Dashboard from './pages/Dashboard.jsx';
import AnalysisLayout from './pages/analysis/AnalysisLayout.jsx';
import Overview from './pages/analysis/Overview.jsx';
import StatModelAnalysis from './pages/analysis/StatModelAnalysis.jsx';
import MlDlAnalysis from './pages/analysis/MlDlAnalysis.jsx';
import ModelComparison from './pages/analysis/ModelComparison.jsx';

function PrivateRoute({ children }) {
  const location = useLocation();
  // 로그인 후 원래 주소(딥링크 쿼리 포함)로 돌아갈 수 있게 from을 넘긴다.
  return hasValidSession() ? children : <Navigate to="/" replace state={{ from: location }} />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          }
        />
        <Route
          path="/analysis"
          element={
            <PrivateRoute>
              <AnalysisLayout />
            </PrivateRoute>
          }
        >
          <Route index element={<Navigate to="overview" replace />} />
          <Route path="overview" element={<Overview />} />
          <Route path="stat" element={<StatModelAnalysis />} />
          <Route path="ml-dl" element={<MlDlAnalysis />} />
          <Route path="comparison" element={<ModelComparison />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
